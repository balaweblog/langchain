import os
from langchain_ollama import ChatOllama
import streamlit as st 

llm =ChatOllama(
    model="gemma:2b")

st.title("Gemma:2b Chatbot")


question = st.text_input('Please enter your question: ')

if question:
    response = llm.invoke(question)
    st.write("Response:", response.content)