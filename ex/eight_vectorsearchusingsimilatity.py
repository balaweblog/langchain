import os
import streamlit as st
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

st.title("Embeddings with Ollama")

llm = ChatOllama(
    model="llama3.2:3b"
)

embeddings = OllamaEmbeddings(model="llama3.2:3b")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "job_listings.txt")

document = TextLoader(file_path).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = text_splitter.split_documents(document)


db = Chroma.from_documents(chunks, embeddings)

query = st.text_input('Search something job related: ')

if query:
    embedding_vector = embeddings.embed_query(query)
    docs = db.similarity_search_by_vector(embedding_vector)
    
    for doc in docs:
        st.write(doc.page_content)


