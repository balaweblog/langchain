import os
from langchain_ollama import ChatOllama
import streamlit as st 

from langchain_core.prompts import PromptTemplate

llm = ChatOllama(
    model="llama3.2:3b")

st.title("Gemma:2b Chatbot")

prompt_template = PromptTemplate(
    input_variables= ["cusinetype", "location", "language"],
    template=""""Find {cusinetype} restaurants {location}. Provide the results in {language}."""
)


cusinetype = st.text_input('Enter the cusinetype')
location = st.text_input('Enter the location')
language = st.text_input('Enter the language')

if language:
    response = llm.invoke(prompt_template.format(cusinetype=cusinetype, location=location,language=language))
    st.write("Response:", response.content)