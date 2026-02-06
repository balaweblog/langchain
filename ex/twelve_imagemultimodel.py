from langchain_core.prompts import ChatPromptTemplate

import base64
import os
import streamlit as st 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

st.title("Image findings with Multimodal LLM")

llm = ChatOllama(
    model="llava"
)


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
question = st.text_input("Enter your question about the image:")

if uploaded_file is not None and question:
    image_data = base64.b64encode(uploaded_file.read()).decode()
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that can describe images."),
            (
                "human",
                [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "low",
                        },
                    },
                ],
            ),
        ]
    ) 

    chain = prompt | llm
    response = chain.invoke({})
    st.write(response)