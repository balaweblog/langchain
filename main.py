import os
from langchain_openai  import ChatOpenAI

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

llm =ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,api_key=OPENAI_API_KEY)


question = input('Please enter your question: ')
response = llm.invoke(question)
print("Response:", response)