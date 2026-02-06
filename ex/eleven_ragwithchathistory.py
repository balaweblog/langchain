
import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

st.title("Product Q&A with RAG and Chat History")

llm = ChatOllama(
    model="llama3.2:3b"
)


embeddings = OllamaEmbeddings(model="llama3.2:3b")


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "product-data.txt")

document = TextLoader(file_path).load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(document)


vectorstoredocument = Chroma.from_documents(chunks, embeddings)

retriever = vectorstoredocument.as_retriever(search_kwargs={"k": 3})



prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context and chat history to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


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
#   2. {chat_history}: Retrieved from session state
#   3. {input}: RunnablePassthrough passes user query unchanged
#   4. prompt_template: Formats context, history, and input into prompt
#   5. llm: Generates response using Ollama model
#   6. StrOutputParser: Extracts text from LLM response
from langchain_core.runnables import RunnableLambda

rag_chain = (
    {
        "context": RunnableLambda(lambda x: x["input"]) | retriever | format_docs,
        "chat_history": RunnableLambda(lambda x: x["chat_history"]),
        "input": RunnableLambda(lambda x: x["input"])
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



st.subheader("Chat History")
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.write(f"**You:** {message.content}")
    elif isinstance(message, AIMessage):
        st.write(f"**Assistant:** {message.content}")


query = st.text_input('Your question')

if query:
    # Prepare chat history as messages for the prompt
    chat_history_list = st.session_state.chat_history
    
    # Invoke RAG chain with context and chat history
    response = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history_list,
        "context": ""  # Will be filled by the retriever in the chain
    })
    
    # Display the response
    st.write(f"**Assistant:** {response}")
    
    # Add to chat history
    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=response))
    
    # Debug: Show retrieved documents
    with st.expander("Retrieved Context"):
        docs = retriever.invoke(query)
        for i, doc in enumerate(docs):
            st.write(f"**Document {i+1}:**\n{doc.page_content}")

# Clear history button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()


