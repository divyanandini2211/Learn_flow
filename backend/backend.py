import os
import shutil
from typing import List, TypedDict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import qdrant_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langgraph.graph import END, StateGraph
from skill_agent import router as skill_agent_router
from planner_agent import router as planner_agent_router
from quiz_agent import router as quiz_agent_router

load_dotenv()

# --- App Setup ---
app = FastAPI()
origins = ["http://localhost", "http://localhost:8501"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(skill_agent_router)
app.include_router(planner_agent_router)
app.include_router(quiz_agent_router)


# --- Environment & Constants ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_COLLECTION_NAME = "student_notes_collection"

if not all([QDRANT_URL, QDRANT_API_KEY, GOOGLE_API_KEY]):
    raise Exception("One or more environment variables are missing")

# --- LangGraph State Definition ---
# This is the "memory" of our agent that gets passed between nodes.
class GraphState(TypedDict):
    question: str
    chat_history: List[tuple[str, str]]
    condensed_question: str
    documents: List[Document]
    generation: str

# --- Models and Tools Initialization ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
vector_store = Qdrant(client=client, collection_name=QDRANT_COLLECTION_NAME, embeddings=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# --- Helper function to format chat history ---
def format_chat_history(chat_history: List[tuple[str, str]]) -> str:
    buffer = ""
    for human, ai in chat_history:
        buffer += f"Human: {human}\nAI: {ai}\n"
    return buffer

# --- LangGraph Nodes ---

def condense_question_node(state):
    """
    The "Memory Saver/Reducer" node.
    It takes the history and the latest question and condenses them
    into a single, clear, standalone question for the retriever.
    """
    print("---NODE: CONDENSE QUESTION (MEMORY REDUCER)---")
    question = state["question"]
    chat_history = state["chat_history"]

    # If there's no history, the question is already standalone.
    if not chat_history:
        return {"condensed_question": question}

    # Prompt to the LLM to rewrite the question
    rewrite_prompt = PromptTemplate.from_template(
        """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
        
        Chat History:
        {chat_history}
        
        Follow Up Input: {question}
        Standalone Question:"""
    )
    
    rewriter_chain = rewrite_prompt | llm | StrOutputParser()
    formatted_history = format_chat_history(chat_history)
    
    condensed_question = rewriter_chain.invoke({
        "chat_history": formatted_history,
        "question": question,
    })
    
    print(f"Original Question: {question}")
    print(f"Condensed Question: {condensed_question}")
    return {"condensed_question": condensed_question}

def retrieve_documents_node(state):
    """Node to retrieve documents from the vector store using the condensed question."""
    print("---NODE: RETRIEVE DOCUMENTS FROM PDF---")
    condensed_question = state["condensed_question"]
    documents = retriever.invoke(condensed_question)
    return {"documents": documents}

def grade_documents_node(state):
    """Node to determine if the retrieved documents are relevant (our fallback logic)."""
    print("---NODE: GRADE DOCUMENT RELEVANCE---")
    question = state["question"] # Grade against the original question for user intent
    documents = state["documents"]
    
    prompt = PromptTemplate.from_template(
        "Based on the retrieved documents, are they relevant to the user's original question? Answer 'yes' or 'no'.\n\nDocuments:\n{documents}\n\nUser Question: {question}"
    )
    grader = prompt | llm | StrOutputParser()
    doc_str = "\n\n".join([d.page_content for d in documents])
    score = grader.invoke({"documents": doc_str, "question": question})
    
    print(f"Relevance Grade: {score}")
    if "yes" in score.lower():
        print("---DECISION: Documents are relevant. Proceed with RAG.---")
        return {"decision": "generate_rag"}
    else:
        print("---DECISION: Documents NOT relevant. Use Gemini API fallback.---")
        return {"decision": "generate_fallback"}

def generate_rag_answer_node(state):
    """Node to generate an answer using the retrieved documents and full history."""
    print("---NODE: GENERATE RAG ANSWER---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]
    
    prompt = PromptTemplate.from_template(
        """You are an expert assistant. Use the chat history and the following retrieved context to answer the user's question.
        \n\nChat History:\n{chat_history}\n\nRetrieved Context:\n{context}\n\nHuman: {question}\nAI:"""
    )
    rag_chain = prompt | llm | StrOutputParser()
    doc_str = "\n\n".join([d.page_content for d in documents])
    formatted_history = format_chat_history(chat_history)
    
    generation = rag_chain.invoke({
        "context": doc_str, "question": question, "chat_history": formatted_history
    })
    return {"generation": generation}

def generate_fallback_answer_node(state):
    """Node to generate an answer using the Gemini API and full history."""
    print("---NODE: GENERATE FALLBACK ANSWER (GEMINI API ONLY)---")
    question = state["question"]
    chat_history = state["chat_history"]
    
    prompt = PromptTemplate.from_template(
        """You are an expert assistant. Answer the user's question based on the chat history and your general knowledge.
        \n\nChat History:\n{chat_history}\n\nHuman: {question}\nAI:"""
    )
    fallback_chain = prompt | llm | StrOutputParser()
    formatted_history = format_chat_history(chat_history)
    generation = fallback_chain.invoke({"question": question, "chat_history": formatted_history})
    return {"generation": generation}

# --- Build and Compile the Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("condense_question", condense_question_node)
workflow.add_node("retrieve", retrieve_documents_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("generate_rag", generate_rag_answer_node)
workflow.add_node("generate_fallback", generate_fallback_answer_node)

workflow.set_entry_point("condense_question")
workflow.add_edge("condense_question", "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", lambda x: x["decision"], {
    "generate_rag": "generate_rag",
    "generate_fallback": "generate_fallback",
})
workflow.add_edge("generate_rag", END)
workflow.add_edge("generate_fallback", END)
app_graph = workflow.compile()

# --- API Layer ---
class Question(BaseModel):
    query: str
    chat_history: list = []

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    temp_file_path = f"./temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        Qdrant.from_documents(
            documents=chunks, embedding=embeddings, url=QDRANT_URL, api_key=QDRANT_API_KEY,
            collection_name=QDRANT_COLLECTION_NAME, force_recreate=True,
        )
        return {"status": "success", "message": f"PDF '{file.filename}' processed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")
    finally:
        if os.path.exists(temp_file_path): os.remove(temp_file_path)
        await file.close()

@app.post("/ask-question")
async def ask_question(request: Question):
    try:
        inputs = {"question": request.query, "chat_history": request.chat_history}
        final_state = app_graph.invoke(inputs)
        return {"answer": final_state["generation"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")