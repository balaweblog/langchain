import os
from langchain_ollama import ChatOllama
import streamlit as st 

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


llm = ChatOllama(
    model="llama3.2:3b")

st.title("Agile Guide")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an Agile Coach. Answer any questions related to Agile Process. Be concise and to the point."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])


chain = prompt | llm 
history_for_chain = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda sessionid: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)

topic = st.text_input('Enter the topic')

if topic:
    response = chain_with_history.invoke({"input": topic}, {"configurable": {"session_id": "default"}})
    st.write("Response:", response.content)

st.write("Chat History:")
for message in history_for_chain.messages:
    st.write(f"{message.type}: {message.content}")