__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
st.title("Duke Housing Assiantant")
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
load_dotenv()
persistent_directory = os.path.join(os.getcwd(), "chroma_db_housing")
local_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

import zipfile

if not os.path.exists("chroma_db_housing/chroma.sqlite3"):
    with zipfile.ZipFile("chroma_db_housing.zip", "r") as zip_ref:
        zip_ref.extractall(".")

if os.path.exists("chroma_db_housing/chroma.sqlite3"):
    st.success("Vectorstore extracted successfully.")
else:
    st.error("Vectorstore not found after extraction.")
# Load environment variables
vectorstore = Chroma(persist_directory=persistent_directory, embedding_function=local_embeddings)
# Initialize Google Gemini AI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a helpful AI Housing assistant of Duke University Students.")]

def query_rag(question: str):
    """Retrieves documents, maintains conversation history, and generates a response."""
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    retrieved_docs = retriever.invoke(question)
    context = ' '.join([f"Source: {doc.metadata['source']}\n{doc.page_content}" for doc in retrieved_docs])
    combined_input = (
        "Here are some documents that might help answer the question: "
        + question
        + "\n\nRelevant Documents:\n"
        + context
        + "\n\nPlease provide a rough answer based only on the provided documents. "
        + "If the answer is not found in the documents, respond with 'I'm not sure'."
        + "If the user asks an open-ended question, respond in detail (100-120 words). If it is a specific question, respond briefly (1-2 sentences). "
        + "At the end of your response, list the sources you used in order of most relevance to least relvance, with proper hyperlinks"
    )
    st.session_state.messages.append(HumanMessage(content=combined_input))
    response = llm.invoke(st.session_state.messages)
    st.session_state.messages.pop()
    st.session_state.messages.append(HumanMessage(question))
    ai_message = AIMessage(content=response.content)
    st.session_state.messages.append(combined_input + ai_message)
    

    return response.content

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(f"{msg.content}")
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(f"{msg.content}")

# React to user input
if prompt := st.chat_input("Ask about duke housing..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = query_rag(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(f"{answer}")
