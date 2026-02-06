import os
from langchain_ollama import ChatOllama
import streamlit as st 
import json

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

llm = ChatOllama(
    model="llama3.2:3b")



llm2 = ChatOllama(
    model="gemma:2b")

st.title("Gemma:2b Chatbot")

first_prompt = PromptTemplate(
    input_variables= ["topic"],
    template=""""You are an experience Speak writer on {topic}. You need to craft a impactful title for a speach on the following topic: {topic}. Answer exactly with one title"""
)

second_prompt = PromptTemplate(
    input_variables= ["title", "emotion"],
    template=""""You are an expert speech writer with the emotion {emotion}. You need to write a detailed speech based on the title: {title}.
    The speech should be written in English and should be engaging and informative.
    For the output with two keys 'title' and 'speach' and fill them with the respective values """
)

first_chain = first_prompt | llm | StrOutputParser()
second_chain = second_prompt | llm2 | JsonOutputParser()
final_chain = first_chain | (lambda title: (st.write(f"Generated Title: {title}"), title)[1]) | (lambda title: second_chain.invoke({"title": title, "emotion": emotion})) 

topic = st.text_input('Enter the topic')
emotion = st.text_input('Enter the emotion')

if topic:
    response = final_chain.invoke({"topic": topic})
    # Normalize to text
    if hasattr(response, "content"):
        resp_text = response.content
    elif isinstance(response, dict) and "content" in response:
        resp_text = response["content"]
    else:
        resp_text = response

    # Try to parse JSON; fallback to stripping Markdown headers and first non-empty line
    parsed = None
    if isinstance(resp_text, (dict, list)):
        parsed = resp_text
    else:
        txt = str(resp_text)
        try:
            parsed = json.loads(txt)
        except Exception:
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if lines:
                first = lines[0]
                # remove leading markdown header markers like '##' or '#'
                first = first.lstrip('#').strip()
                parsed = first
            else:
                parsed = txt

    st.write("Response:", parsed)