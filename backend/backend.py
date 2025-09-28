import os
import shutil
from typing import List, TypedDict, Literal
from functools import partial

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import qdrant_client
from qdrant_client.http import models
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langgraph.graph import END, StateGraph
# Your agent routers
from skill_agent import router as skill_agent_router
from planner_agent import router as planner_agent_router
from quiz_agent import router as quiz_agent_router
# CORRECTLY import your Wikipedia agent functions
from wikipedia_agent import get_tool_router_node, run_wikipedia_node

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
# UPDATED: Added 'source' to track the decision from the router
class GraphState(TypedDict):
    question: str
    chat_history: List[tuple[str, str]]
    condensed_question: str
    documents: List[Document]
    generation: str
    source: Literal["vectorstore", "wikipedia", "fallback"]

# --- Models and Tools Initialization ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

try:
    client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
    print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists.")
except Exception:
    print(f"Collection '{QDRANT_COLLECTION_NAME}' not found. Creating new collection.")
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )
    print("Collection created successfully.")

vector_store = Qdrant(client=client, collection_name=QDRANT_COLLECTION_NAME, embeddings=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

def format_chat_history(chat_history: List[tuple[str, str]]) -> str:
    buffer = ""
    for human, ai in chat_history:
        buffer += f"Human: {human}\nAI: {ai}\n"
    return buffer

# --- LangGraph Nodes (Your original logic is preserved) ---

def condense_question_node(state):
    print("---NODE: CONDENSE QUESTION (MEMORY REDUCER)---")
    question = state["question"]
    chat_history = state["chat_history"]
    if not chat_history:
        return {"condensed_question": question}
    rewrite_prompt = PromptTemplate.from_template(
        "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\n\nFollow Up Input: {question}\nStandalone Question:"
    )
    rewriter_chain = rewrite_prompt | llm | StrOutputParser()
    formatted_history = format_chat_history(chat_history)
    condensed_question = rewriter_chain.invoke({"chat_history": formatted_history, "question": question})
    print(f"Condensed Question: {condensed_question}")
    return {"condensed_question": condensed_question}

def retrieve_documents_node(state):
    print("---NODE: RETRIEVE DOCUMENTS FROM PDF---")
    condensed_question = state["condensed_question"]
    documents = retriever.invoke(condensed_question)
    return {"documents": documents}

def grade_documents_node(state):
    print("---NODE: GRADE DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    if not documents:
        print("---DECISION: No documents found. Using Fallback.---")
        return {"source": "fallback"} # Changed to update source, not decision
    prompt = PromptTemplate.from_template(
        "Based on the retrieved documents, are they relevant to the user's original question? Answer 'yes' or 'no'.\n\nDocuments:\n{documents}\n\nUser Question: {question}"
    )
    grader = prompt | llm | StrOutputParser()
    doc_str = "\n\n".join([d.page_content for d in documents])
    score = grader.invoke({"documents": doc_str, "question": question})
    if "yes" in score.lower():
        print("---DECISION: Documents are relevant. Proceed with RAG.---")
        return {"source": "vectorstore"} # Keep source as vectorstore
    else:
        print("---DECISION: Documents NOT relevant. Use Gemini API fallback.---")
        return {"source": "fallback"} # Change source to fallback

def generate_answer_node(state):
    """
    A single node to generate an answer. It checks for a source URL in the metadata
    and MANUALLY appends it to the LLM's response to guarantee it is included.
    """
    print(f"---NODE: GENERATE ANSWER (SOURCE: {state['source']})---")
    question = state["question"]
    documents = state.get("documents", [])
    chat_history = state["chat_history"]
    formatted_history = format_chat_history(chat_history)

    # First, generate the base answer from the LLM
    if documents:
        print("---INFO: Generating answer with context from documents.---")
        doc_str = "\n\n".join([d.page_content for d in documents])
        prompt = PromptTemplate.from_template(
            """You are an expert assistant. Use the chat history and the following retrieved context to answer the user's question.
            Do not mention the source in your response.

            Chat History:\n{chat_history}
            Retrieved Context:\n{context}
            Human: {question}
            AI:"""
        )
        chain = prompt | llm | StrOutputParser()
        base_answer = chain.invoke({"context": doc_str, "question": question, "chat_history": formatted_history})
    else: # Fallback case
        print("---INFO: Generating fallback answer with general knowledge.---")
        prompt = PromptTemplate.from_template(
            "You are an expert assistant. Answer the user's question based on the chat history and your general knowledge.\n\nChat History:\n{chat_history}\n\nHuman: {question}\nAI:"
        )
        chain = prompt | llm | StrOutputParser()
        base_answer = chain.invoke({"question": question, "chat_history": formatted_history})

    # --- THIS IS THE FIX ---
    # Now, check for a source URL and append it to the generated answer ourselves.
    final_generation = base_answer
    if documents and documents[0].metadata.get("source_url"):
        source_url = documents[0].metadata["source_url"]
        print(f"---INFO: Appending source URL to the answer: {source_url}---")
        final_generation += f"\n\nSource: {source_url}"
    # -----------------------
        
    return {"generation": final_generation}

# --- Build and Compile the INTEGRATED Graph ---
workflow = StateGraph(GraphState)

# Create instances of the tool nodes using the factory
tool_router_node = get_tool_router_node(llm)
# Use partial to pass the llm argument to the wikipedia node
wikipedia_node_with_llm = partial(run_wikipedia_node, llm=llm)

# Add all nodes to the graph
workflow.add_node("condense_question", condense_question_node)
workflow.add_node("tool_router", tool_router_node)
workflow.add_node("retrieve_pdf", retrieve_documents_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("run_wikipedia", wikipedia_node_with_llm)
workflow.add_node("generate_answer", generate_answer_node)

# Define the graph's structure
workflow.set_entry_point("condense_question")
workflow.add_edge("condense_question", "tool_router")

# Conditional routing after the tool router
workflow.add_conditional_edges(
    "tool_router",
    lambda x: x["source"],
    {
        "vectorstore": "retrieve_pdf",
        "wikipedia": "run_wikipedia",
        "fallback": "generate_answer", # Go directly to generation if fallback
    }
)

workflow.add_edge("retrieve_pdf", "grade_documents")

# Conditional routing after grading the PDF documents
workflow.add_conditional_edges(
    "grade_documents",
    lambda x: x["source"],
    {
        "vectorstore": "generate_answer", # If relevant, generate with context
        "fallback": "generate_answer",    # If not relevant, generate without context
    }
)

workflow.add_edge("run_wikipedia", "generate_answer") # After wikipedia, generate with context
workflow.add_edge("generate_answer", END)

# Compile the final graph
app_graph = workflow.compile()


# --- API Layer (Your original code is preserved) ---
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
            documents=chunks,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=QDRANT_COLLECTION_NAME,
            force_recreate=True
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
        # The compiled graph is invoked here
        final_state = app_graph.invoke(inputs)
        return {"answer": final_state["generation"]}
    except Exception as e:
        # This will now give more specific errors if the graph fails
        print(f"Error invoking graph: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred in the agent graph: {e}")