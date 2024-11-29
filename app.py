import streamlit as st

from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from utils import MEMORY
from agent import load_agent


st.set_page_config(page_title="Langchain Research Assistant",
                   page_icon=":robot:")


st.header("Ask your question Padawan!")

strategy = st.radio(
    "Reasoning Strategy",
    ("plan-and-solve", "zero-shot-react"),
)

tool_names = st.multiselect(
    "Which tools would you like to use?",
    [
        "critical_search",
        "llm-math",
        "ddg-search",
        "wolfram_alpha",
        "google_search",
        "wikipedia",
        "arxiv",
        "python_repl",
    ],
    ["ddg-search", "wolfram_alpha", "wikipedia"]
)

if st.sidebar.button("Clear message history"):
    MEMORY.chat_memory.clear()
    
avatars = {
    "human": "user",
    "ai": "assistant"
}
for msg in MEMORY.chat_memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)
    
agent_chain = load_agent(tool_names=tool_names, strategy=strategy)

if prompt := st.chat_input(placeholder="Ask me anything!!"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_chain.invoke(
            {"input": prompt} ,
            {"callbacks": [st_callback]}
        )
        st.write(response["output"])
