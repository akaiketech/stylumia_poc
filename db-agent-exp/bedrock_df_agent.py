from typing import Dict, Generator, List, Tuple
from abc import ABC, abstractmethod
from loguru import logger
import pandas as pd
from langsmith import traceable
from IPython.display import display, Markdown
from tools.sql_tool import MySQlEngine, get_sql_lc_tool
import json
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# from agent_planner import OpenAIAgentPlanner, AgentStep, AgentAction, AgentFinish
from bedrock_agent_planner import (
    BedrockClaudeAgentPlanner,
    AgentStep,
    AgentAction,
    AgentFinish,
    AgentMessage,
    MultiToolAgentAction,
    MultiToolAgentStep,
    ToolResult,
)
from pathlib import Path
from tools.python_tool import get_python_lc_tool, get_matching_values_tool
from tools.plot_generator import get_plot_gen_lc_tool
from PIL import Image
from pydantic.v1 import BaseModel, Field
from langchain_aws import ChatBedrock
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from io import BytesIO


class BaseGlossaryRetriever(ABC):

    @abstractmethod
    def query(self, query: str) -> List[str]:
        pass


# TODO: Remove columns which are null
class TableVectorRetriever(ABC):

    @abstractmethod
    def get_relevant_tables(self, query: str, k: int = 10) -> List[Dict[str, str]]:
        """
        Return list of table dictionaries with following keys:
        - name
        - description
        - score
        """
        pass

    @abstractmethod
    def get_relevant_columns(
        self, table_name: str, query: str, k: int = 30
    ) -> List[Dict[str, str]]:
        """
        Return list of column dictionaries with following keys:
        - name
        - description
        - score

        # Approach
        - if total columns < k, return all columns
        - Get all primary keys
        - Do not consider null columns
        - Select top k similar columns
        """
        pass


class DataFrameMetadataVectorRetriever(TableVectorRetriever):

    def __init__(self, data_config: List[Tuple[pd.DataFrame, Dict]]):
        self.data_config = data_config
        self.table_store = None
        self.column_store = None
        self.prepare()

    def prepare(self):
        table_docs = []
        column_docs = []
        for df, df_metadata in self.data_config:
            table_name = df_metadata.get("name")
            table_doc = Document(
                page_content=f"{table_name}: {df_metadata.get('description')}",
                metadata={
                    "table_desc": df_metadata.get("description"),
                    "table_name": table_name,
                },
            )
            table_docs.append(table_doc)
            for col_meta in df_metadata["columns"]:
                column_doc = Document(
                    page_content=f"{col_meta.get('name')}:{col_meta.get('data_type')}:{col_meta.get('description')}",
                    metadata={
                        "column_desc": col_meta.get("description"),
                        "table_name": table_name,
                        "column_name": col_meta.get("name"),
                        "column_type": col_meta.get("data_type"),
                    },
                )
                column_docs.append(column_doc)

        self.table_store = FAISS.from_documents(
            table_docs, OpenAIEmbeddings(model="text-embedding-3-large")
        )
        self.column_store = FAISS.from_documents(
            column_docs, OpenAIEmbeddings(model="text-embedding-3-large")
        )

    def get_relevant_tables(self, query: str, k: int = 10) -> List[Dict[str, str]]:
        result = self.table_store.similarity_search_with_score(query, k=k)
        return [
            {
                "table_name": t[0].metadata["table_name"],
                "table_desc": t[0].metadata["table_desc"],
                "score": t[1],
            }
            for t in result
        ]

    def get_relevant_columns(
        self, table_name: str, query: str, k: int = 30
    ) -> List[Dict[str, str]]:
        result = self.column_store.similarity_search_with_score(
            query, k=k, filter={"table_name": table_name}
        )
        return [
            {
                "column_name": t[0].metadata["column_name"],
                "column_type": t[0].metadata["column_type"],
                "column_desc": t[0].metadata["column_desc"],
                "score": t[1],
            }
            for t in result
        ]


class FollowUpResponse(BaseModel):
    is_follow_up: bool
    rephrased_question: str = Field(
        description="rephrase the question from the user's perspective to be a standalone question so that history is not required,pay close attention to previous user messages so that the question is not diluted/becomes vague"
    )
    history_summary: str = Field(
        description="summarize and keep only the relevant information from message history that will be useful for the new repharased question, make sure to include all the relevant terms, numbers & tables which will be usefull"
    )


claude_prompt = """
You are an AI assistant tasked with analyzing conversation history and a new user input to generate a structured response. Here's the information you'll be working with:

<message_history>
{{MESSAGE_HISTORY}}
</message_history>

<new_user_input>
{{NEW_USER_INPUT}}
</new_user_input>

Your task is to analyze this information and produce a JSON output with three keys: is_follow_up, rephrased_question, and history_summary. Follow these steps:

1. Determine if the new user input is a follow-up question:
   - Carefully examine the new user input and the message history.
   - Set is_follow_up to true if the new input refers to or builds upon previous messages.
   - Set is_follow_up to false if the new input appears to be a new, unrelated question.

2. Rephrase the question:
   - Create a standalone version of the question from the user's perspective.
   - Incorporate any context from previous messages that's necessary to understand the question.
   - Ensure the rephrased question is clear and complete, without requiring additional context to understand.

3. Summarize relevant history:
   - Review the message history and identify information relevant to the rephrased question.
   - Include key terms, numbers, and any tables that provide context or are necessary to answer the question.
   - Summarize this information concisely, focusing only on details that directly relate to the rephrased question.

4. Format your response as a JSON object with the following structure:
   {
     "is_follow_up": boolean,
     "rephrased_question": "Your rephrased question here",
     "history_summary": "Your summary of relevant history here"
   }

Ensure that your JSON is properly formatted and that all values are appropriate for their respective keys (boolean for is_follow_up, strings for the other two).

Provide your final output within <response> tags."""


def get_history_prompt(messages):
    prompt = ""
    for m in messages:
        role = "user"
        if m["role"] == "assistant":
            role = "assistant"
        prompt += f'{role}:{m["content"]}\n'
    return prompt


@traceable
def identify_followup(message_history, query):
    followup_system_prompt = """
     Previous message history
    >>>
    {conv_history}
    <<<
    
    identify if the new user input is a followup
    
    return response using tool
    """

    followup_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", followup_system_prompt),
            ("human", "new input: {query}"),
        ]
    )

    follow_up_chain = followup_prompt_template | ChatOpenAI(
       model="gpt-4o",temperature=0
    ).with_structured_output(FollowUpResponse)
    follow_result = follow_up_chain.invoke(
        {"query": query, "conv_history": get_history_prompt(message_history)}
    )
    return follow_result


def get_clarification_history(messages):
    relevant_history = []
    for m in messages[::-1]:
        if m["answer_state"] != "clarification":
            break
        relevant_history.append(m)
    return relevant_history[::-1]


class ClarifyQuestionResponse(BaseModel):
    clarification_required: bool
    follow_up_question: str | None = Field(
        description="if clarification is required, then what follow-up question to ask to resolve ambiguity?"
    )
    rephrased_user_question: str | None = Field(
        description="if ambiguity is resolved, then reframe the complete question"
    )
    suggested_questions: List[str] = Field(
        description="if clarification is required, suggestion some sample questions to help the user"
    )
    reason: str = Field(description="Why is clarification required?")


@traceable
def clarify_question(
    question: str,
    message_history: List,
    glossary: List[str],
    tables: List[Dict[str, str]] | None = None,
    history_summary: str = None,
    domain: str = None,
) -> Dict:
    clarify_system_prompt = """
    You are an AI analyst helping stakeholders with answers to specific questions on datasets.
    Your job is to identify if the questions is clear enough to be able to provide an answer.
    
    Assume that is user is not stupid, he knows what he is asking
    
    When is clarification required?
    >>>
    * Only when the question is very vague (Ignore small ambiguity)
    <<<
    
    When is clarification not required?
    >>>
    * Ignore small ambiguity
    * User intent is clear
    * User is asking pointed question
    * It is ok to make monir assumptions (don't ask every little detail)
    * if the user has resolved the confusion (look at history & new use response)
    <<<

    ### Key Definitions:
    ** Lay of the land**: This referes to the assortment mix of all the retailers in the dataset
    **Assortment Across an Attribute:** This refers to the distribution of products across categories within a specific attribute. When asked about the assortment, generate a pivot table with the attribute categories as the index and retailer names as the columns. The values should represent the count of items in each category. Additionally, reset the index so that the attribute categories appear as a standard column in the DataFrame.
    
    **Assortment Across key Attributes:** Generate the individual assortments across the attributes - Type, Color Changing, Power Source and concatenate all the pivot tables and provide me a concatenated data **Assortment Across Categories:** Generate the individual assortments across the Categorical columns - Type, Color Changing, Power Source and concatenate all the pivot tables and provide me a concatenated data

    **Top Performing attributes:** This refers to the attribute categories where retailer top seller (%) is greater than or equal to the retailer supply(%) of the attribute category
    To achieve this filter the rows based on the condition. Retailer supply(%) of the attribute category is the proportion of products belong to the attribute category. Do this only when the question is regarding Top Performing attributes
    **Under Performing attributes:** This refers to the attribute categories where retailer top seller (%) is less than the retailer supply(%) of the attribute category
    To achieve this filter the rows based on the condition. Retailer supply(%) of the attribute category is the proportion of products belong to the attribute category. Do this only when the question is regarding Top Performing attributes
    **Under Indexed**:
    These are the attributes where the retailer has to expand their volume. To identify these, get the common top performing attributes in the market & the retailer.The idea is to identify the attributes where both the market,  retailer are doing well. You would need to check the top seller percent of market vis a vis our supply
    **Over Indexed**:
    These are the attributes where the retailer has to refresh/reduce their volume. To identify these, get the common attributes among Market's **Top Seller** attributes & the retailer's Under Performing attributes. The idea is to identify the attributes where the market is doing well but not the retailer.
    You would need to check the top seller percent of market vis a vis our supply
    ** Where to expand**:
    This is same as the attributes that are over indexed.
    ** Where to refresh/reduce**:
    This is same as the attributes that are under indexed.
    ** Where to monitor**:
    The common attributes among Retailer's Top performing attributes & the Market's Under Performing attributes. The idea is to identify the attributes where the retailer is doing well but not the market.
    ** Where to rationalise**:
    The common attributes among Retailer's Under performing attributes & the Market's Under Performing attributes. The idea is to identify the attributes where both the retailer,  market are not doing well
    ### Ask questions for more clarity:
    - Ask the required questions if the provided information is insufficient for answering
    
    return response using tool
    """

    user_prompt = """
    History Summary
    >>>
    {history_summary}
    <<<

    The following tables are retrieved based on the similarity search from the database
    >>>
    {table_info}
    <<<


    History
    >>>
    {chat_history}
    <<<
    New User Input: {user_input}
    """

    clarify_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", clarify_system_prompt),
            ("human", user_prompt),
        ]
    )
    clarify_chain = clarify_prompt_template | ChatOpenAI(
       model="gpt-4o",temperature=0
    ).with_structured_output(ClarifyQuestionResponse)

    clarification_relevant_history = get_clarification_history(message_history)
    clarification_query_question = "\n".join(
        [m["content"] for m in clarification_relevant_history if m["role"] == "human"]
        + [question]
    )
    prompt_history = get_history_prompt(clarification_relevant_history)
    prompt_similar_tables = "\n".join(
        [f"{d['table_name']}:{d['table_desc']}" for d in tables]
    )
    table_prompt = get_db_table_prompt(tables)
    history_summary = history_summary or ""
    result: ClarifyQuestionResponse = clarify_chain.invoke(
        {
            "user_input": clarification_query_question,
            "chat_history": prompt_history,
            "table_info": table_prompt,
            "history_summary": history_summary,
        }
    )
    return {
        "clarification_required": result.clarification_required,
        "follow_up_question": result.follow_up_question,
        "rephrased_user_question": result.rephrased_user_question,
        "suggested_questions": result.suggested_questions,
        "metadata": {
            "glossary": glossary,
            "tables": tables,
            "domain": domain,
            "clarification_reasoning": result.reason,
        },
    }


def get_db_table_prompt(relevant_table_columns):
    table_prompts = []
    for table_meta in relevant_table_columns:
        table_name = table_meta["table_name"]
        relevant_columns = table_meta["column_data"]
        schema = (
            f"{table_name} (\n"
            + "\n".join(
                [
                    f"{d['column_name']}:{d['column_type']} {d['column_desc']}"
                    for d in relevant_columns
                ]
            )
            + ")"
        )
        table_prompt = f"Table Name: {table_name}\nDescription: {table_meta['table_desc']}\nSchema:\n{schema}"
        table_prompts.append(table_prompt)
    return "\n----\n".join(table_prompts)


class PlanResponse(BaseModel):
    high_level_approach: str = Field(
        description="high level approach to solve the question"
    )
    assumptions: List[str] = Field(
        description="assumptions made while solving the question"
    )
    steps: List[str] = Field(description="steps to solve the question")
    rejected: bool = Field(description="is the plan rejected?")
    rejection_reason: str = Field(description="reason for rejection")


@traceable
def generate_plan(
    question: str,
    relevant_table_columns: List[Dict[str, str]],
    glossary: List[str] = None,
) -> Dict:
    """
    Generate the execution plan for the question

    # Approach
    - similar to `clarify_question`
    - use plan_hint
    """
    plan_system_prompt = """
    You have access to below tools
    
    sql_workbench: execute sql query
    python_env: execute python code, SQL table output is automatically loaded in  python_env as a pandas dataframe
    plot_generator: generates plots for the dataframes generated in python_env
    
    In the the db you have access to the following tables:
    {table_info}
    
    
    Given a question create a high level approach, assumptions and list of steps on detailing the approach, make sure to put backticks for columns & tables
    
    Assumption Instruction
    ----------------
    * As part of the assumption include user intent in a more analytical form (removing ambiguity if present)
    
    Plan Instructions:
    ----------------
    * As part of the instruction mention the tools that you will be using
    * As part of the steps, avoid generic words like analyse, instead give more details on techniques that we should use to analyze
    * Try to make relevant plots for the user if that helps the question
    
    output structure is a json as
    {{
    "high_level_approach": "", 
    assumptions: [""], 
    steps: [""],
    rejected: boolean,
    rejection_reason: ""
    }}
    """

    plan_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", plan_system_prompt),
            ("human", "{question}"),
        ]
    )

    plan_chain = plan_prompt_template | ChatOpenAI(
       model="gpt-4o",temperature=0
    ).with_structured_output(PlanResponse)

    table_prompt = get_db_table_prompt(relevant_table_columns)
    return plan_chain.invoke({"table_info": table_prompt, "question": question})


def create_plan_prompt(plan_data):
    assumption_text = "* " + "\n* ".join(plan_data.assumptions)
    steps_text = "* " + "\n* ".join(plan_data.steps)
    return f"High Level Plan:\n{plan_data.high_level_approach}\n\nAssumptions:\n{assumption_text}\n\nApproximate Steps:\n{steps_text}"


class FileAgent:
    def __init__(
        self,
        data_config: List[Tuple[pd.DataFrame, Dict]],
        plot_folder: Path,
        prev_messages=None,
        clarify_question: bool = True,
        pre_generate_plan: bool = True,
        glossary_retriever: BaseGlossaryRetriever | None = None,
        db_retriever: TableVectorRetriever | None = None,
        init_state: Dict | None = None,
        max_steps: int = 20,
    ):
        self.data_config = data_config
        self.db_retriever = db_retriever
        self.plot_folder = plot_folder
        self.prev_messages = prev_messages or []
        self.clarify_question = clarify_question
        self.pre_generate_plan = pre_generate_plan
        self.glossary_retriever = glossary_retriever
        self.state = init_state or {}
        self.max_steps = max_steps

    def _identify_followup(self, question):
        return identify_followup(self.prev_messages, question)

    def _clarify_question(self, question: str) -> Dict:
        glossary = (
            self.glossary_retriever.query(question)
            if self.glossary_retriever is not None
            else []
        )
        glossary_text = "\n".join(glossary)
        query_text = f"question:{question}\nGlossary:{glossary_text}"
        # relevant_tables = (
        #     self.db_retriever.get_relevant_tables(query_text)
        #     if self.db_retriever is not None
        #     else []
        # )
        relevant_table_columns = self.get_relevant_table_and_columns(question)

        return clarify_question(
            question,
            self.prev_messages,
            glossary,
            relevant_table_columns,
            history_summary=self.state["follow_up_data"].history_summary,
        )

    def get_relevant_table_and_columns(self, query: str):
        relevant_tables = (
            self.db_retriever.get_relevant_tables(query)
            if self.db_retriever is not None
            else []
        )
        result = []
        for table_meta in relevant_tables:
            table_name = table_meta["table_name"]
            relevant_columns = self.db_retriever.get_relevant_columns(table_name, query)
            result.append(
                {
                    "table_name": table_name,
                    "table_desc": table_meta["table_desc"],
                    "column_data": relevant_columns,
                }
            )
        return result

    def _generate_plan(self, question: str) -> str:
        relevant_table_columns = self.get_relevant_table_and_columns(question)
        glossary = (
            self.glossary_retriever.query(question)
            if self.glossary_retriever is not None
            else []
        )
        return generate_plan(question, relevant_table_columns, glossary)

    @traceable
    def generate_streaming_response(self, question: str) -> Generator:
        question = f"I am a retailer named A and need help with analysis on assortment related data\nQuestion: {question}"
        self.state["status"] = "processing"
        self.state["original_question"] = question
        self.state["question"] = question

        yield {
            "event": "identifying_followup",
            "status": "processing",
        }
        follow_up_data = self._identify_followup(self.state["question"])
        self.state["follow_up_data"] = follow_up_data

        if follow_up_data.is_follow_up:
            self.state["question"] = follow_up_data.rephrased_question
            yield {
                "event": "identified_as_followup",
                "status": "processing",
                "rephrased_question": follow_up_data.rephrased_question,
            }

        if self.clarify_question:
            yield {
                "event": "check_clarification",
                "status": "processing",
            }
            clarification_question_data = self._clarify_question(self.state["question"])
            self.state["clarification_question_data"] = clarification_question_data
            if clarification_question_data["clarification_required"]:
                self.state["status"] = "complete"
                self.state["content"] = clarification_question_data[
                    "follow_up_question"
                ]
                self.state["answer_state"] = "clarification"
                yield {
                    "event": "final_response",
                    "clarification_required": True,
                    "final_response": self.state["content"],
                }
                return
            self.state["question"] = clarification_question_data[
                "rephrased_user_question"
            ]
            yield {
                "event": "question_rephrased",
                "status": "processing",
                "rephrased_question": clarification_question_data[
                    "rephrased_user_question"
                ],
            }

        if self.pre_generate_plan:
            yield {
                "event": "generating_plan",
                "status": "processing",
            }
            plan_data = self._generate_plan(self.state["question"])
            self.state["plan_data"] = plan_data
            yield {"event": "plan_generated", "status": "processing", "plan": plan_data}

            if plan_data.rejected:
                self.state["status"] = "complete"
                self.state["content"] = plan_data.rejection_reason
                yield {
                    "event": "final_response",
                    "status": "complete",
                    "question_rejected": True,
                    "final_response": plan_data.rejection_reason,
                }
                return

        yield from self.run_agent()

    def prepare_agent(self) -> None:
        agent_system_prompt = """
        You are a data analyst agent designed to interact with a SQL database and python to output meaningful insights for the question asked by the stakeholder.
        
        
        Python Environment Tool Intructions:
        >>>
        * plotting libraries is not present in the python environment, use the plot_generator tool instead to generate plots
        * Never manually create dataframe
        * Do not use matplotlib, plotly, etc to create the plot, use plot_generator tool instead
        * Never try to print the whole dataframe if it can be huge, instead print the head, tail or sample of the dataframe to check the data
        * Never use disk operations to save or load data, for pandas dataframes use the data already loaded in memory
        * avoid using print statements, instead in the python code list the dataframe/variable at the end of the code (it will get printed automatically)
        * Do not import standard libraries, assume those are already imported
        <<<

        Matching Values Tool Instructions:
        >>>
        * Returns the most similar values in column based on the input string
        * If a filter on  dataframe returns zero results, then use this tool to find the right filter for the string column
        * Prefer This function over pandas.Series.str.contains pandas function
        <<<
        
        Plot generator Tool Instructions:
        >>>
        * Exclusively use this function to create plot, do not use Python environment
        * Create visualization whenever possible to support the answer
        <<<

        Final Response Intruction:
        >>>
        * Always Give final answer as tables (markdown format) which consists of both percentage and numerical answers
        * Whenever top seller is used in calculation, mention it in the explaination
        * Make sure the headers you give for tables are related to retailer business (eg. SKU Count instead of count)
        * Keep the tables before textual analysis
        * Be Brief and clear in the response, use markdown
        * Never give recommendataions,or areas of improvements until asked 
        <<<

        Data Approach Instruction
        >>>
        ### Key Definitions:
        ** Lay of the land**: This referes to the assortment mix of all the retailers in the dataset of Type attribute 
        **Assortment Across an Attribute:** This refers to the distribution of products across categories within a specific attribute. When asked about the assortment, generate a pivot table with the attribute categories as the index and retailer names as the columns. The values should represent the count of items in each category. Additionally, reset the index so that the attribute categories appear as a standard column in the DataFrame.
        For each retailer, create two additional columns:
        Percentage of Total Products: This column should reflect the percentage of the retailer's overall assortment that falls within each attribute category.
        Percentage of Top Seller Products: This column should show the percentage of top-selling products within each attribute category, relative to the retailer's total assortment.
        To achieve this, first create a pivot table containing only the top-selling products for each category within that attribute.
        Then, calculate the percentage of these top-selling products in each category over all the top selling products of the retailer.
        top_seller_percentage = top_seller_count_of_category_of_retailer/total_top_seller_products_of_retailer

        #### Follow the below code snippet for reference to acheive this
        df_a = df[df['Retailer'] == "A"]
        # Create pivot tables for each attribute
        type_pivot = df_a.groupby('Type').size().reset_index(name="A")
        # Calculate the percentage of total products for each attribute category
        type_pivot["A (%)"] = (type_pivot["A"] / df_a.shape[0]) * 100
        # Filter the dataframe for 'A' and 'Top Seller' products
        df_a_top_seller = df_a[df_a['Top Seller'] == 'Yes']
        # Create pivot tables for each attribute
        type_pivot_top_seller = df_a_top_seller.groupby('Type').size().reset_index(name="Top Seller A")
        # Calculate the percentage of top seller products for each attribute category
        type_pivot_top_seller["Top Seller A (%)"] = (type_pivot_top_seller["Top Seller A"] / df_a_top_seller.shape[0]) * 100
        Ensure that the output includes the count, percentage of total products, and percentage of top seller products for each retailer.

        **Assortment Across key Attributes:** Generate the individual assortments across the attributes - Type, Color Changing, Power Source and concatenate all the pivot tables and provide me a concatenated data
        **Assortment Across Categories:** Generate the individual assortments across the categorical columns - Type, Color Changing, Power Source and concatenate
        all the pivot tables and provide me a concatenated data
        **Top Performing attributes:** This refers to the attribute categories where retailer top seller (%) is greater than or equal to the retailer supply(%) of the attribute category. To achieve this filter the rows based on the condition. Retailer supply(%) of the attribute category is the proportion of products belong to the attribute category. Do this only when the question is regarding Top Performing attributes
        **Under Performing attributes:** This refers to the attribute categories where retailer top seller (%) is less than the retailer supply(%) of the attribute category. To achieve this filter the rows based on the condition. Retailer supply(%) of the attribute category is the proportion of products belong to the attribute category. Do this only when the question is regarding Top Performing attributes
        **refresh**: attributes where we have supply similar to market but market top seller is way more ahead
        **rightly Indexed**: These are attributes where retailer is at par volume compared to the market, performance on this metric is evaluated on top sellers
        **Under Indexed**:These are the attributes where the retailer has to expand their volume based on the whether the attribute is common among the top performing attributes in the market & the retailer. To identify these, get the common top performing attributes in the market & the retailer.The idea is to identify the attributes where both the market,  retailer are doing well.
        You would need to check the top seller percent of market vis a vis our supply
        **Over Indexed**:These are the attributes where the retailer has to refresh/reduce their volume based on the whether the attribute is common among the top performing attributes in the market & the under performing attributes for the retailer. To identify these, get the common attributes among Market's Top performing attributes & the retailer's Under Performing attributes. The idea is to identify the attributes where the market is doing well but not the retailer.
        You would need to check the top seller percent of market vis a vis our supply
        ** Where to expand**: This is same as the attributes that are over indexed.
        ** Where to refresh/reduce**: This is same as the attributes that are under indexed.
        ** Where to monitor**: The common attributes among Retailer's Top performing attributes & the Market's Under Performing attributes. The idea is to identify the attributes where the retailer is doing well but not the market.
        ** Where to rationalise**: The common attributes among Retailer's Under performing attributes & the Market's Under Performing attributes. The idea is to identify the attributes where both the retailer,  market are not doing well
    

        ### Ask questions for more clarity:Ask the required questions if the provided information is insufficient for answering

        
        <<<
        Other Important Instructions
        >>>
        * Your Primary attribute is `Type`, make sure to always use this only, until and  unless a different attribute is specified.
        * Always give full table if the number of rows is less than 15
        * In case question is for over and under indexing make sure to compare for top selling products and mention it in your response
        * Never ever give potential areas of improvements or recommendations , till asked
            * If specifically asked then refrain from using exact figures, use top sellers percent as benchmark and avoid speculative suggestion
            * always give directional advice instead of something like (increase sku by 7-10 items/10 percent increase)
        * Incase of a general question do 2 analysis one for A(me) and one for the market
        * avoid giving generic responses
        * Plot charts whereever possible. Always run relevant python code before calling the plotting tool
        * Always round the results to 2 decimal places in intermediate and final steps
    

        <<<
        Table Info:
        ========
        {table_info}

`
        Python environment State
        ```
        {python_code_history}
        ```
        
        If the question does not seem related to the data, just return "I don't know" as the answer.
        """
        human_input = """
Previus Conv History Summary:
>>>
{conv_hist_summary}
<<<

Question:{input}"""
        if self.pre_generate_plan:
            human_input += "\n>>>{approach}\n<<<"

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", agent_system_prompt),
                ("human", human_input),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        relevant_tables_columns = self.get_relevant_table_and_columns(
            self.state["question"]
        )
        table_info = get_db_table_prompt(relevant_tables_columns)

        py_tool = get_python_lc_tool(self.plot_folder)
        match_tool = get_matching_values_tool(py_tool)
        self.py_tool_name = py_tool.name
        plot_tool = get_plot_gen_lc_tool(
            source_python_tool=py_tool, plot_folder=self.plot_folder, with_query=False
        )
        table_name2df = {df_meta["name"]: df for df, df_meta in self.data_config}
        for i, table_meta in enumerate(relevant_tables_columns):
            code = f"#Load {table_meta['table_name']}\ndf{i} = pd.read_parquet('data/{table_meta['table_name']}.parquet')\n"
            py_tool.func.__self__.update_execution_history(code)
            py_tool.func.__self__.locals[f"df{i}"] = table_name2df[
                table_meta["table_name"]
            ]

        self.tools = [py_tool, match_tool, plot_tool]
        self.tool_name2tool = {t.name: t for t in self.tools}
        conv_hist_summary = self.state["follow_up_data"].history_summary

        if self.pre_generate_plan:
            approach = create_plan_prompt(self.state["plan_data"])
            prompt_template = prompt_template.partial(
                table_info=table_info,
                approach=approach.strip(),
                conv_hist_summary=conv_hist_summary,
            )
        else:
            prompt_template = prompt_template.partial(
                table_info=table_info,
                conv_hist_summary=conv_hist_summary,
            )
        self.agent_planner = BedrockClaudeAgentPlanner(
            prompt_template, "anthropic.claude-3-5-sonnet-20240620-v1:0", self.tools
        )
        self.state["raw_intermediate_steps"] = []
        self.state["intermediate_steps"] = []

    def prepare_intermediate_step(self):
        """
        Update ` self.state['intermediate_steps']`
        - Trim useless step
        - Summarize older steps if the data is too much
        """
        # return self.state['raw_intermediate_steps']
        # remove all the steps whose's .metadata.get("error") is True except the last step (always keep the last step)
        return self.state["raw_intermediate_steps"]
        # steps = self.state["raw_intermediate_steps"]
        # if len(steps) == 0:
        #     return steps
        # new_steps = []
        # for step in steps[:-1]:
        #     if not hasattr(step, "metadata"):
        #         new_steps.append(step)
        #         continue
        #     if not step.metadata.get("error"):
        #         new_steps.append(step)
        # new_steps.append(steps[-1])
        # return new_steps

    @traceable
    def step_agent(self):
        python_code_history = self.tool_name2tool[
            self.py_tool_name
        ].func.__self__.execution_history
        result = self.agent_planner.plan(
            self.prepare_intermediate_step(),
            input=self.state["question"],
            python_code_history=python_code_history,
        )
        if isinstance(result, AgentFinish):
            return result

        if not isinstance(result, MultiToolAgentAction):
            raise ValueError("Only MultiToolAgentAction is supported in action")

        tool_results = []
        for tool_action in result.tool_actions:
            tool = self.tool_name2tool[tool_action.tool]
            tool_result = tool.invoke(tool_action.tool_input)
            status = (
                "success"
                if not tool_result.get("metadata", {}).get("error")
                else "error"
            )
            observation_content = {"text": tool_result.get("observation")}
            if tool.name == "plot_generator":
                try:
                    
                    with open(tool_result.get("metadata")["image_path"], "rb") as f:
                        
                        image_data = f.read()
                    observation_content = {
                        "image": {"format": "jpeg", "source": {"bytes": image_data}}
                    }
                    fig = tool_result.get("metadata")["fig"]
                except KeyError as e:
                    logger.error(f"Error in getting image data: {e}")
                    print(tool_result)
            tool_results.append(
                ToolResult(
                    tool_action=tool_action,
                    content=observation_content,
                    metadata=tool_result.get("metadata"),
                    
                    status=status,
                )
            )

        agent_step = MultiToolAgentStep(action=result, tool_results=tool_results)
        return agent_step

    def run_agent(self) -> Generator:
        yield {
            "event": "preparing_agent",
            "status": "processing",
        }
        self.prepare_agent()
        yield {
            "event": "agent_ready",
            "status": "processing",
        }

        for _ in range(self.max_steps):
            step_output = self.step_agent()
            if isinstance(step_output, AgentFinish):
                self.state["content"] = step_output.log
                yield {
                    "event": "final_response",
                    "status": "complete",
                    "final_response": step_output.log,
                }
                return
            for message in step_output.action.message_log:
                yield {
                    "event": "intermediate_step",
                    "status": "processing",
                    "message": message.content,
                    "message_type": "text",
                }
            for tool_result in step_output.tool_results:
                yield {
                    "event": "intermediate_step",
                    "status": "processing",
                    "tool": tool_result.tool_action.tool,
                    "tool_input": tool_result.tool_action.tool_input,
                    "step_response": tool_result.content,
                    "message_type": "tool_output",
                    "metadata": tool_result.metadata,
                }
            self.state["raw_intermediate_steps"].append(step_output)
        else:
            logger.warning("Max steps reached")
