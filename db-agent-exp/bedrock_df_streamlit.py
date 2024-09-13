import streamlit as st
import pandas as pd
import json
from bedrock_df_agent import DataFrameMetadataVectorRetriever, FileAgent
from PIL import Image
from typing import Dict
from io import BytesIO
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
import os

load_dotenv("../.env")
set_llm_cache(SQLiteCache(database_path="./temp_folder/llm_cache.sqlite.db"))

# file = open("../.env")
# st.write(file.read())
# file.close()

SUPPORTED_TABLES = [
    "outdoor_lighting_products_renamed_zipcode"
    # "customer_support",
    # "product_category_sku_analysis",
    # "stylumia_all_data"
]

st.set_page_config(layout="wide")


@st.cache_resource
def load_table_data(selected_tables):
    data_config = []
    for table_path_name in selected_tables:
        df = pd.read_parquet(f"data/processed_data/{table_path_name}.parquet")
        with open(f"data/processed_data/{table_path_name}_metadata.json") as f:
            df_metadata = json.load(f)
        data_config.append((df, df_metadata))
    table_retriever = DataFrameMetadataVectorRetriever(data_config)
    return data_config, table_retriever


def main():
    st.title("Analytics Chat App")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.write("Table Selection")
        selected_tables = st.multiselect(
            "Select Table", SUPPORTED_TABLES, default=SUPPORTED_TABLES[0]
        )

        clear_chat = st.button("Clear Chat")
        if clear_chat:
            st.session_state.messages = []

    if not selected_tables:
        return

    data_config, table_retriever = load_table_data(selected_tables)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question:")

    
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                agent = FileAgent(
                    data_config,
                    Path("./temp_folder"),
                    prev_messages=st.session_state.messages,
                    db_retriever=table_retriever,
                    pre_generate_plan=False,
                )
                clarification_required = False
                for event in agent.generate_streaming_response(prompt):
                    if event.get("event") == "identifying_followup":
                        st.toast("Identifying Followup")
                    elif event.get("event") == "rephrasing_question":
                        st.toast("Rephrasing Question")
                    elif event.get("event") == "identified_as_followup":
                        st.toast("Identified As Followup")
                        st.info("Question rephrased: " + event["rephrased_question"])
                    elif event.get("event") == "check_clarification":
                        st.toast("Check Clarification")
                    elif event.get("event") == "question_rephrased":
                        st.toast("Clarification not required")
                    elif event.get("event") == "preparing_agent":
                        st.toast("Preparing Agent")
                    elif event.get("event") == "agent_ready":
                        st.toast("Agent Ready", icon="✅")
                    elif event.get("event") == "final_response":
                        st.markdown(event["final_response"])
                        if event.get("clarification_required"):
                            st.markdown("> **Clarification Required ⚠️**")
                    elif event.get("event") == "intermediate_step":
                        if "message" in event:
                            for message in event["message"]:
                                if message["type"] == "text":
                                    st.write(message["text"])
                                # elif message["type"] == "tool_use":
                                #     with st.expander(message["name"]):
                                #         st.write(message["input"])
                        elif "tool" in event:
                            with st.expander(event["tool"]):
                                if event["tool"] == "python_env":
                                    st.markdown("### Code:")
                                    st.code(event["tool_input"]["code"])
                                    st.markdown("### Output:")
                                    if "table" in event.get("metadata", {}):
                                        st.caption(
                                            f'```\n{event["metadata"].get("str_log")}\n```'
                                        )
                                        st.dataframe(event["metadata"]["table"])
                                    else:
                                        st.caption(
                                            f'```\n{event["step_response"].get("text")}\n```'
                                        )
                                    st.json(event, expanded=False)
                                elif event["tool"] == "plot_generator":
                                    st.markdown("### Plot Instruction:")
                                    st.markdown(event["tool_input"]["plot_instruction"])
                                    if not event.get("metadata", {}).get("error"):
                                        image_data = event["step_response"]["image"][
                                            "source"
                                        ]["bytes"]
                                        st.image(Image.open(BytesIO(image_data)))
                                    else:
                                        st.caption(
                                            f'```\n{event["step_response"].get("text")}\n```'
                                        )
                                else:
                                    st.write(event)
                        else:
                            st.write(event)

                    else:
                        st.write(event)

                st.session_state.messages.append(
                    {
                        "role": "human",
                        "content": prompt,
                        "answer_state": agent.state.get("answer_state"),
                    }
                )
                st.session_state.messages.append(
                    {
                        "role": "ai",
                        "content": agent.state["content"],
                        "answer_state": agent.state.get("answer_state"),
                    }
                )
                # st.write(st.session_state.messages)  : Need to reduce the size of input going into it

if __name__ == "__main__":
    main()
