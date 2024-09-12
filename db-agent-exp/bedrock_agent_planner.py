from langchain_aws import ChatBedrock
from pydantic.v1 import BaseModel
from typing import Union, List, Sequence, Literal
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    BaseMessage,
    ToolMessage,
)


class AgentFinish(BaseModel):
    log: str


class AgentMessage(BaseModel):
    message: str


class AgentAction(BaseModel):
    tool_id: str
    tool: str
    tool_input: Union[dict, str]
    log: str
    message_log: Sequence[BaseMessage]


class AgentParseError(BaseModel):
    log: str


class ToolAction(BaseModel):
    tool_id: str
    tool: str
    tool_input: Union[dict, str]


class ToolResult(BaseModel):
    tool_action: ToolAction
    content: dict
    metadata: dict | None
    status: Literal["success", "error"]


class MultiToolAgentAction(BaseModel):
    tool_actions: List[ToolAction]
    message_log: Sequence[BaseMessage]


class AgentStep(BaseModel):
    action: AgentAction
    observation: str
    metadata: dict | None


class MultiToolAgentStep(BaseModel):
    action: MultiToolAgentAction
    tool_results: List[ToolResult]


# def bedrock_agent_parser(ai_message):
#     if ai_message.response_metadata["stopReason"] == "end_turn":
#         return AgentFinish(log=ai_message.content)
#     if ai_message.response_metadata["stopReason"] != "tool_use":
#         raise AgentParseError(
#             log=f"{ai_message.response_metadata['stopReason']} is not supported"
#         )

#     steps = []
#     for message in ai_message.content:
#         if message["type"] == "text":
#             steps.append(AgentMessage(message=message["text"]))
#         elif message["type"] == "tool_use":
#             steps.append(
#                 AgentAction(
#                     tool_id=message["id"],
#                     tool=message["name"],
#                     tool_input=message["input"],
#                     log="",
#                     message_log=[],
#                 )
#             )
#     return steps


def bedrock_agent_parser(ai_message):
    if ai_message.response_metadata["stopReason"] == "end_turn":
        return AgentFinish(log=ai_message.content)
    if ai_message.response_metadata["stopReason"] != "tool_use":
        raise AgentParseError(
            log=f"{ai_message.response_metadata['stopReason']} is not supported"
        )

    tool_actions = []
    for message in ai_message.content:
        if message["type"] == "tool_use":
            tool_actions.append(
                ToolAction(
                    tool_id=message["id"],
                    tool=message["name"],
                    tool_input=message["input"],
                )
            )

    return MultiToolAgentAction(tool_actions=tool_actions, message_log=[ai_message])


# class BedrockClaudeAgentPlanner:
#     def __init__(self, prompt, model_name, tools):
#         self.prompt = prompt
#         self.model_name = model_name
#         self.tools = tools
#         self.tool_name2tool = {t.name: t for t in tools}
#         self.planner_chain = (
#             self.prompt
#             | ChatBedrock(
#                 model_id=self.model_name,
#                 model_kwargs=dict(temperature=0),
#                 beta_use_converse_api=True,
#             ).bind_tools(tools)
#             | bedrock_agent_parser
#         )

#     def _prepare_intermediate_steps(self, intermediate_steps):
#         messages = []
#         for step in intermediate_steps:
#             if isinstance(step, AgentStep):
#                 messages.append(
#                     ToolMessage(
#                         tool_call_id=step.action.tool_id,
#                         content=step.observation,
#                     )
#                 )
#             elif isinstance(step, AgentAction):
#                 messages.append(
#                     AIMessage(
#                         content=[
#                             {
#                                 "type": "tool_use",
#                                 "id": step.tool_id,
#                                 "name": step.tool,
#                                 "input": step.tool_input,
#                             }
#                         ]
#                     )
#                 )
#             else:
#                 print(type(step))
#                 messages.append(AIMessage(content=step.message))
#         return messages

#     def plan(
#         self,
#         intermediate_steps: List[AgentStep | AgentFinish | AgentParseError] | None,
#         **kwargs: dict,
#     ):
#         if intermediate_steps is None:
#             intermediate_steps = []
#         inputs = {
#             **kwargs,
#             "agent_scratchpad": self._prepare_intermediate_steps(intermediate_steps),
#         }
#         return self.planner_chain.invoke(inputs)


class BedrockClaudeAgentPlanner:
    def __init__(self, prompt, model_name, tools):
        self.prompt = prompt
        self.model_name = model_name
        self.tools = tools
        self.tool_name2tool = {t.name: t for t in tools}
        self.planner_chain = (
            self.prompt
            | ChatBedrock(
                model_id=self.model_name,
                model_kwargs=dict(temperature=0),
                beta_use_converse_api=True,
            ).bind_tools(tools)
            | bedrock_agent_parser
        )

    def _prepare_intermediate_steps(self, intermediate_steps: List[MultiToolAgentStep]):
        messages = []
        for step in intermediate_steps:
            if not isinstance(step, MultiToolAgentStep):
                raise ValueError(
                    "All intermediate steps for should be of type MultiToolAgentStep"
                )
            messages.extend(step.action.message_log)
            tool_messages = []
            for tool_result in step.tool_results:
                tool_messages.append(
                    {
                        "toolResult": {
                            "content": [tool_result.content],
                            "status": tool_result.status,
                            "toolUseId": tool_result.tool_action.tool_id,
                        }
                    }
                )
            messages.append(HumanMessage(content=tool_messages))
        return messages

    def plan(
        self, intermediate_steps: List[MultiToolAgentStep] | None, **kwargs: dict
    ) -> MultiToolAgentAction | AgentFinish:
        if intermediate_steps is None:
            intermediate_steps = []
        inputs = {
            **kwargs,
            "agent_scratchpad": self._prepare_intermediate_steps(intermediate_steps),
        }
        return self.planner_chain.invoke(inputs)
