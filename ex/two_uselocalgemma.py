import os
from langchain_ollama import ChatOllama

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

llm =ChatOllama(
    model="gemma:2b")


question = input('Please enter your question: ')
response = llm.invoke(question)
print("Response:", response)