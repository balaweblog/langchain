"""
RAG (Retrieval-Augmented Generation) Application
================================================
This application implements a RAG system using LangChain, Ollama, and Streamlit.

Architecture:
1. Document Loading: Reads product-data.txt file
2. Text Splitting: Splits documents into chunks for better retrieval
3. Embeddings: Uses Ollama embeddings to convert text to vectors
4. Vector Store: Stores embeddings in Chroma for similarity search
5. Retrieval: Fetches top-k relevant documents for user queries
6. Generation: Uses LLM to answer questions based on retrieved context

Flow: Query → Retriever → Format Docs → Prompt → LLM → Response
"""

import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.title("Product Q&A with RAG")

llm = ChatOllama(
    model="llama3.2:3b"
)


embeddings = OllamaEmbeddings(model="llama3.2:3b")


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "product-data.txt")

document = TextLoader(file_path).load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(document)

# ==================== Vector Store Setup ====================
# Create Chroma vector store from document chunks
# Each chunk is embedded using OllamaEmbeddings and stored for similarity search
vectorstoredocument = Chroma.from_documents(chunks, embeddings)

# Create retriever that returns top-3 most relevant documents
# search_kwargs={"k": 3}: Retrieve 3 most similar documents for context
retriever = vectorstoredocument.as_retriever(search_kwargs={"k": 3})



# ==================== Prompt Template ====================
# Defines the interaction between user query and LLM
# {context}: Will be filled with retrieved documents
# {input}: Will be filled with user's question
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise.\n\n{context}"),
    ("human", "{input}")
])


# ==================== Helper Function ====================
def format_docs(docs):
    """
    Formats retrieved documents into a readable string.
    Joins multiple documents with double newlines for clarity.
    """
    return "\n\n".join(doc.page_content for doc in docs)


# ==================== RAG Chain Assembly ====================
# Uses LangChain Expression Language (LCEL) with pipe operator
# Chain flow:
#   1. {context}: retriever fetches docs, format_docs converts to string
#   2. {input}: RunnablePassthrough passes user query unchanged
#   3. prompt_template: Formats context and input into prompt
#   4. llm: Generates response using Ollama model
#   5. StrOutputParser: Extracts text from LLM response
rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)


# ==================== Streamlit UI ====================
# User input text field for questions
query = st.text_input('Your question')

if query:
    response = rag_chain.invoke(query)
    st.write("Answer:", response)
    
    # Debug: Show retrieved documents
    with st.expander("Retrieved Context"):
        docs = retriever.invoke(query)
        for i, doc in enumerate(docs):
            st.write(f"**Document {i+1}:**\n{doc.page_content}")
    st.write("Answer:", response)


