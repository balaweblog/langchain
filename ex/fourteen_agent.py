import os
import streamlit as st
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.globals import set_debug

set_debug(True)

# 1. LLM setup

llm = ChatOllama(
    model="llama3.2:3b"
)


# 2. Tools (Wikipedia + DuckDuckGo)
tools = load_tools(["wikipedia", "ddg-search"])  # Load tools as a list to avoid iterating characters


# 3. ReAct-style system prompt
react_system_prompt = """
You are a ReAct-style AI agent.
Follow this loop carefully:
1. THOUGHT: Think step by step about what to do next.
2. ACTION: When needed, call one of the tools (wikipedia, ddg-search).
3. OBSERVATION: Read the tool result and decide the next step.
Repeat THOUGHT → ACTION → OBSERVATION
until you are ready to give the final answer.

When you are confident, stop using tools and respond with a clear, concise final answer to the user.
"""


# 4. Create Agent (new v1 API)

#TODO: Create Agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=react_system_prompt)

# 5. Streamlit UI
st.title("AI Agent (ReAct style – LangChain v1)")

task = st.text_input("Assign me a task")

if task:
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": task}
            ]
        }
    )
    final_answer = result['messages'][-1]
    st.write(final_answer.content)