# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING, Optional

import litellm
from datarobot_genai.core.agents import InvokeReturn, make_system_prompt
from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.chat import agent_chat_completion_wrapper
from datarobot_genai.core.mcp import MCPConfig
from datarobot_genai.langgraph.agent import datarobot_agent_class_from_langgraph
from datarobot_genai.langgraph.llm import get_llm
from datarobot_genai.langgraph.mcp import mcp_tools_context
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from openai.types.chat import CompletionCreateParams

if TYPE_CHECKING:
    from ragas import MultiTurnSample

litellm.modify_params = True

_PLACEHOLDER_MODELS = frozenset({"unknown"})

DEFAULT_DATASET_ID = "6a234c26fbe8f6deb1434646"
DEFAULT_DEPLOYMENT_ID = "6a2355338f421c2e8b54b99e"

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant for ERCOT electricity market DAM (Day-Ahead Market) price forecasting. "
            "You help users obtain 24-hour price predictions using DataRobot deployed ML models. "
            "Chat history is provided via {chat_history} (it may be empty). "
            "Use it when helpful to stay consistent across turns.",
        ),
        (
            "user",
            "{topic}",
        ),
    ]
)


def graph_factory(
    llm: BaseChatModel, tools: list[BaseTool], verbose: bool = False
) -> StateGraph[MessagesState]:
    agent_coordinator = create_agent(
        llm,
        tools=[],
        system_prompt=make_system_prompt(
            "You are a coordinator agent for ERCOT DAM price forecasting.\n"
            "\n"
            "Your task is to interpret the user's request and extract the following parameters:\n"
            f"- dataset_id (default: {DEFAULT_DATASET_ID})\n"
            f"- deployment_id (default: {DEFAULT_DEPLOYMENT_ID})\n"
            "- hub_name (e.g. HB_HOUSTON)\n"
            "- forecast_range_start (datetime in 'YYYY-MM-DD HH:MM:SS' format, UTC)\n"
            "- forecast_range_end (datetime in 'YYYY-MM-DD HH:MM:SS' format, UTC)\n"
            "\n"
            "If any parameter is missing from the user request, use the default values above.\n"
            "Output a clear structured summary of the extracted parameters, then instruct the "
            "Forecaster to proceed with the prediction using these parameters.\n"
            "\n"
            "Do NOT call any tools yourself. Just extract parameters and pass instructions to the next agent.",
        ),
        name="coordinator_agent",
        debug=verbose,
    )

    agent_forecaster = create_agent(
        llm,
        tools=tools,
        system_prompt=make_system_prompt(
            "You are a forecaster agent for ERCOT DAM price prediction.\n"
            "\n"
            "You have access to the following DataRobot MCP tools:\n"
            "- get_dataset_details(dataset_id): Retrieve details of a DataRobot dataset\n"
            "- predict_realtime(dataset_id, deployment_id, forecast_range_start, forecast_range_end): "
            "Execute prediction using a DataRobot deployment\n"
            "\n"
            "Steps to follow:\n"
            "1. Call get_dataset_details(dataset_id=<dataset_id>) to confirm the dataset is accessible.\n"
            "2. Call predict_realtime(\n"
            "     dataset_id=<dataset_id>,\n"
            "     deployment_id=<deployment_id>,\n"
            "     forecast_range_start=<forecast_range_start>,\n"
            "     forecast_range_end=<forecast_range_end>\n"
            "   ) to obtain the 24-hour DAM price predictions.\n"
            "3. Return the full prediction results (timestamps and predicted DAM prices) to the next agent.\n"
            "\n"
            "Authentication is already handled by the MCP server. "
            "If a tool call fails, report the error clearly so the reporter can inform the user.",
        ),
        name="forecaster_agent",
        debug=verbose,
    )

    agent_reporter = create_agent(
        llm,
        tools=[],
        system_prompt=make_system_prompt(
            "You are a reporter agent that presents ERCOT DAM price forecast results to business decision makers.\n"
            "\n"
            "Based on the prediction results from the forecaster, produce a concise Japanese-language report "
            "in Markdown format that includes:\n"
            "\n"
            "1. **予測サマリー**: 期間・対象ハブ・予測時間数\n"
            "2. **価格統計**: 最高値・最安値・平均価格（USD/MWh）とそれぞれの発生時刻\n"
            "3. **時間帯分析**: 朝（6-12時）・昼（12-18時）・夜（18-24時）・深夜（0-6時）の平均価格傾向\n"
            "4. **ビジネスインサイト**: 電力調達・販売の観点から意思決定者に有益な2〜3点のインサイト\n"
            "5. **DataRobotモデル情報**: 使用したデプロイID・データセットID\n"
            "\n"
            "Keep the report clear, data-driven, and actionable for a C-level audience. "
            "All numeric values should be rounded to 2 decimal places.",
        ),
        name="reporter_agent",
        debug=verbose,
    )

    def relay(state: MessagesState) -> MessagesState:
        last = state["messages"][-1]
        if isinstance(last, AIMessage):
            return {"messages": [HumanMessage(content=last.content)]}
        return {"messages": []}

    workflow = StateGraph(MessagesState)
    workflow.add_node("coordinator_node", agent_coordinator)
    workflow.add_node("coordinator_to_forecaster", relay)
    workflow.add_node("forecaster_node", agent_forecaster)
    workflow.add_node("forecaster_to_reporter", relay)
    workflow.add_node("reporter_node", agent_reporter)

    workflow.add_edge(START, "coordinator_node")
    workflow.add_edge("coordinator_node", "coordinator_to_forecaster")
    workflow.add_edge("coordinator_to_forecaster", "forecaster_node")
    workflow.add_edge("forecaster_node", "forecaster_to_reporter")
    workflow.add_edge("forecaster_to_reporter", "reporter_node")
    workflow.add_edge("reporter_node", END)

    return workflow


MyAgent = datarobot_agent_class_from_langgraph(graph_factory, prompt_template)


async def custompy_adaptor(
    completion_create_params: CompletionCreateParams,
) -> InvokeReturn | tuple[str, Optional["MultiTurnSample"], UsageMetrics]:
    forwarded_headers = completion_create_params.get("forwarded_headers", {})
    authorization_context = completion_create_params.get("authorization_context", {})
    mcp_config = MCPConfig(
        forwarded_headers=forwarded_headers,
        authorization_context=authorization_context,
    )
    mcp_tools_factory = lambda: mcp_tools_context(mcp_config)  # noqa: E731
    model_name = completion_create_params.get("model")
    agent = MyAgent(
        llm=get_llm(
            model_name=model_name if model_name not in _PLACEHOLDER_MODELS else None
        ),
        verbose=completion_create_params.get("verbose", True),  # type: ignore[arg-type]
        timeout=completion_create_params.get("timeout", 300),  # type: ignore[arg-type]
        forwarded_headers=forwarded_headers,  # type: ignore[arg-type]
    )
    return await agent_chat_completion_wrapper(
        agent, completion_create_params, mcp_tools_factory
    )
