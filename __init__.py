
from langchain.chains import LLMChain

from langchain_openai import ChatOpenAI
from langchain_experimental import plan_and_execute

from typing import Literal

from tool_loader import load_tools

ReasoningStrategy = Literal["zero_shot_react", "plan_and_solve"]

def load_agent(
    tool_names: list[str],
    strategy: ReasoningStrategy = "zero_shot_react"
) -> LLMChain:
    """
    """
    llm = ChatOpenAI(temperature=0, streaming=True)
    tools = load_tools(tool_names, llm=llm)
    
    if strategy == "plan_and_solve":
        pl