import os
from langchain_ollama import ChatOllama
import streamlit as st 

from langchain_core.prompts import PromptTemplate

llm = ChatOllama(
    model="llama3.2:3b")

st.title("Gemma:2b Chatbot")

prompt_template = PromptTemplate(
    input_variables= ["cusinetype", "location", "language"],
    template=""""Find {cusinetype} restaurants {location}. and rating of {rating} Provide the results in {language}."""
)


cusinetype = st.text_input('Enter the cusinetype')
location = st.text_input('Enter the location')
language = st.text_input('Enter the language')
rating = st.selectbox("Select minimum rating", ["1","2","3","4","5"])

chain = prompt_template | llm

if language:
    response = chain.invoke({
        "cusinetype": cusinetype,
        "location": location,
        "language": language,
        "rating": rating
    })
    st.write("Response:", response.content)