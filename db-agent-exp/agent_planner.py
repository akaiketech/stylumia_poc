from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from typing import Union, List, Sequence
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from pydantic.v1 import BaseModel
import json


class AgentFinish(BaseModel):
    log: str


class AgentAction(BaseModel):
    tool_id: str
    tool: str
    tool_input: Union[dict, str]
    log: str
    message_log: Sequence[BaseMessage]


class AgentParseError(BaseModel):
    log: str


class AgentStep(BaseModel):
    action: AgentAction
    observation: str
    metadata: dict | None


def openai_agent_parser(ai_message):
    if not ai_message.additional_kwargs.get("tool_calls"):
        return AgentFinish(log=ai_message.content)

    actions = []
    for tool_call in ai_message.additional_kwargs["tool_calls"]:
        tool_name = tool_call["function"]["name"]
        try:
            tool_input = json.loads(tool_call["function"]["arguments"])
            actions.append(
                AgentAction(
                    tool_id=tool_call["id"],
                    tool=tool_name,
                    tool_input=tool_input,
                    log=ai_message.content,
                    message_log=[ai_message]
                )
            )
        except json.JSONDecodeError:
            actions.append(
                AgentParseError(
                    log=f"Could not parse tool input: {tool_name} because the arguments is not a valid JSON."
                )
            )

    return actions


class OpenAIAgentPlanner:
    def __init__(self, prompt, model_name, tools):
        llm = ChatOpenAI(model=model_name, temperature=0)
        self.tools = tools
        self.llm_with_tools = llm.bind(
            tools=[convert_to_openai_tool(tool) for tool in tools]
        )
        self.prompt = prompt
        self.chain = self.prompt | self.llm_with_tools | openai_agent_parser

    def _prepare_intermediate_steps(
        self, intermediate_steps: List[AgentStep | AgentFinish | AgentParseError]
    ) -> List[BaseMessage]:
        messages = []
        for step in intermediate_steps:
            new_messages = []
            if isinstance(step, AgentStep):
                new_messages += step.action.message_log
                new_messages.append(
                    ToolMessage(
                        tool_call_id=step.action.tool_id,
                        content=step.observation,
                        additional_kwargs={"name": step.action.tool},
                    )
                )
            else:
                new_messages.append(AIMessage(content=step.log))
            # Ensure duplicate messages are not added (can happen for multi tool calls)
            messages.extend([new for new in new_messages if new not in messages])
        return messages

    def plan(
        self,
        intermediate_steps: List[AgentStep | AgentFinish | AgentParseError]
        | None = None,
        **kwargs: dict,
    ) -> AgentAction | AgentFinish | AgentParseError:
        if intermediate_steps is None:
            intermediate_steps = []
        inputs = {
            **kwargs,
            "agent_scratchpad": self._prepare_intermediate_steps(intermediate_steps),
        }
        return self.chain.invoke(inputs)
