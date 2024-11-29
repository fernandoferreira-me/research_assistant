from langchain import hub

from langchain.chains import LLMChain
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_experimental.plan_and_execute.planners.chat_planner import load_chat_planner
from langchain_experimental.plan_and_execute.executors.agent_executor import load_agent_executor
from langchain_experimental.plan_and_execute import PlanAndExecute
from typing import Literal

from tool_loader import load_tools

ReasoningStrategy = Literal["zero_shot_react", "plan_and_solve"]

def load_agent(
    tool_names: list[str],
    strategy: ReasoningStrategy = "zero_shot_react"
) -> LLMChain:
    """
    """
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", verbose=True)
    tools = load_tools(tool_names, llm=llm)
    if strategy == "plan_and_solve":
        planner = load_chat_planner(llm, output_format="Thought and Action")
        executor = load_agent_executor(
            llm,
            tools,
            output_format="Thought and Action",
            handle_parsing_errors=True,
            verbose=True
        )
        return PlanAndExecute(planner=planner, executor=executor, verbose=True)
    
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)
    return AgentExecutor(agent=agent,
                         tools=tools,
                         handle_parsing_errors=True,
                         verbose=True)
