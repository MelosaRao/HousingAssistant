import gradio as gr
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load vectorstore
persistent_directory = os.path.join(os.getcwd(), "chroma_db_housing")
local_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(persist_directory=persistent_directory, embedding_function=local_embeddings)

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Initialize chat history
system_prompt = SystemMessage(content="You are a helpful AI Housing assistant of Duke University Students.")
chat_history = [system_prompt]

# Core RAG function
def query_rag(user_input, history):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    retrieved_docs = retriever.invoke(user_input)
    context = ' '.join([f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" for doc in retrieved_docs])

    combined_input = (
        "Here are some documents that might help answer the question: "
        + user_input
        + "\n\nRelevant Documents:\n"
        + context
        + "\n\nPlease provide a rough answer based only on the provided documents. "
        + "If the answer is not found in the documents, respond with 'I'm not sure'."
        + "If the user asks an open-ended question, respond in detail (100-120 words). If it is a specific question, respond briefly (1-2 sentences). "
        + "At the end of your response, list the sources you used in order of most relevance to least relevance, with proper hyperlinks."
    )

    # Maintain conversation context
    chat_history.append(HumanMessage(content=combined_input))
    response = llm.invoke(chat_history)
    chat_history.append(HumanMessage(user_input))
    chat_history.append(AIMessage(content=response.content))

    return response.content

# Gradio Chat Interface
chatbot = gr.ChatInterface(fn=query_rag, title="Duke Housing Assistant")

if __name__ == "__main__":
    chatbot.launch()