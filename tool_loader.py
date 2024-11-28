from langchain.agents import (AgentExecutor,
                              Tool,
                              create_self_ask_with_search_agent)
from langchain.chains import LLMMathChain

# Tools
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.wolfram_alpha import WolframAlphaQueryRun
from langchain_community.tools.google_search import GoogleSearchRun
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

# Utilities
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_experimental.tools import PythonREPLTool

from langchain import hub

from typing import Optional

python_repl = PythonREPLTool()

def load_tools(
    tool_names: list[str],
    llm: Optional[BaseLanguageModel] = None
) -> list[BaseTool]:
    """
    """
    prompt = hub.pull("hwchase17/self-ask-with-search")
    search_wrapper = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())
    search_tool = Tool(
        name="Intermediate Answer",
        func=search_wrapper.invoke,
        description="Search"
    )
    self_ask_agent = AgentExecutor(
        agent=create_self_ask_with_search_agent(
            prompt=prompt,
            llm=llm,
            tools=[search_tool]),
        tools = [search_tool]
    )
    
    available_tools = {
        "ddg-search": DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper()),
        "wolfram_alpha": WolframAlphaQueryRun(api_wrapper=WolframAlphaAPIWrapper()),
        "google_search": GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper()),
        "wikipedia": WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        "arxiv": ArxivQueryRun(api_wrapper=ArxivAPIWrapper()),
        "python_repl": Tool(
            name="python repl",
            description="""
            A Python shell. Use this to execute Python commands.
            Input should be a valid python command. If you want to see the output of a command, use the print function.
            `print(...)`
            """,
            func=python_repl.run
        ),
        "critical_search": Tool.from_function(
            func=self_ask_agent.invoke,
            name="Self-ask agent",
            description="""
            A tool to answer complicated questions.
            Useful for when you need to answer questions about current events.
            
            Input should be a question.
            """,
        )
    }
    
    tool = []
    for name in tool_names:
        tool.append(available_tools[name])
        
    return tool