import streamlit as st
from langchain_ollama import ChatOllama

llm = ChatOllama(model="deepseek-r1:8b")

st.title("DeepSeek — Simple Consumer")

query = st.text_input("Enter your question for DeepSeek")

if query:
    response = llm.invoke(query)
    # show text content if available
    content = getattr(response, "content", None) or getattr(response, "text", None) or str(response)
    st.write(content)