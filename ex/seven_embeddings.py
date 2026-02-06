import os
import streamlit as st
import numpy as np
from langchain_ollama import OllamaEmbeddings

st.title("Embeddings with Ollama")

embeddings = OllamaEmbeddings(
    model="llama3.2:3b"
)

question = st.text_input('Please enter your question: ')
question1 = st.text_input('Please enter your question 1: ')


if question and question1:
    response = embeddings.embed_query(question)
    response1 = embeddings.embed_query(question1)
    similarity_score = np.dot(response, response1)
    st.write("Similarity Score:")
    st.write(similarity_score)