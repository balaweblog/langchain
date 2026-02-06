import os
from langchain_ollama import ChatOllama
import streamlit as st 

from langchain_core.prompts import PromptTemplate

llm = ChatOllama(
    model="gemma:2b")

st.title("Gemma:2b Chatbot")

prompt_template = PromptTemplate(
    input_variables=["country", "language"],
    template=""""You are expert in traditional cuisines. You provide information
    about fictional places. if you know the answer give the results. if not say I dont know 
    Example format is Answer the question: what is the traditional cuisine of {country} and
     answer in short paras in  {language}."""
)


country = st.text_input('Enter the country')
language = st.text_input('Enter the language')

if language:
    response = llm.invoke(prompt_template.format(country=country, language=language))
    st.write("Response:", response.content)