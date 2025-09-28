import os
import shutil
from typing import List, TypedDict
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
from skill_agent import router as skill_agent_router
from planner_agent import router as planner_agent_router
from quiz_agent import router as quiz_agent_router
# Import the new Wikipedia agent components
from wikipedia_agent import get_tool_router_node, run_wikipedia_node


load_dotenv()

# --- App Setup ---
app = FastAPI()
origins = ["http://localhost", "http://localhost:8501"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=[""], allow_headers=[""])
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
    # Add keys for routing and grading decisions
    source: str
    decision: str

# --- Models and Tools Initialization ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# --- Auto-create collection on startup ---
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
# -----------------------------------------

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
    """Condenses the history and latest question into a single, standalone question."""
    print("---NODE: CONDENSE QUESTION---")
    question = state["question"]
    chat_history = state["chat_history"]

    if not chat_history:
        print("---DECISION: No chat history, using original question.---")
        return {"condensed_question": question}

    rewrite_prompt = PromptTemplate.from_template(
        """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
        Chat History: {chat_history}
        Follow Up Input: {question}
        Standalone Question:"""
    )
    rewriter_chain = rewrite_prompt | llm | StrOutputParser()
    condensed_question = rewriter_chain.invoke({
        "chat_history": format_chat_history(chat_history), "question": question
    })
    print(f"Condensed Question: {condensed_question}")
    return {"condensed_question": condensed_question}

def retrieve_documents_node(state):
    """Retrieves documents from the vector store."""
    print("---NODE: RETRIEVE DOCUMENTS FROM VECTORSTORE---")
    condensed_question = state["condensed_question"]
    documents = retriever.invoke(condensed_question)
    return {"documents": documents}

def grade_documents_node(state):
    """Determines if the retrieved documents are relevant to the question."""
    print("---NODE: GRADE DOCUMENT RELEVANCE---")
    question = state["condensed_question"]
    documents = state["documents"]

    if not documents:
        print("---DECISION: No documents found, routing to fallback.---")
        return {"decision": "generate_fallback"}

    prompt = PromptTemplate.from_template(
        """You are a grader. Based on the retrieved documents, are they relevant to the user's question? Answer 'yes' or 'no'.
        Documents: {documents}
        User Question: {question}"""
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
    """Generates an answer using the retrieved documents."""
    print("---NODE: GENERATE RAG ANSWER---")
    question = state["condensed_question"]
    documents = state["documents"]
    
    # Check if the source is Wikipedia to use a different, more detailed prompt
    source = "vectorstore" # Default source
    if documents:
        source = documents[0].metadata.get("source", "vectorstore")
    
    if source == "wikipedia":
        prompt_template = """You are an AI assistant. Your task is to provide a comprehensive summary of the following text to answer the user's question.
        Do not be overly concise. Capture the main points, key details, and important context from the provided text.

        Context:
        {context}
        
        User Question: {question}
        
        Comprehensive Summary:"""
    else:
        prompt_template = """You are an AI study assistant. Use the following retrieved context to answer the user's question. If you don't know the answer from the context, say that you don't know.
        Be concise and helpful.

        Context: {context}

        Question: {question}"""

    prompt = PromptTemplate.from_template(prompt_template)
    rag_chain = prompt | llm | StrOutputParser()
    doc_str = "\n\n".join([d.page_content for d in documents])
    generation = rag_chain.invoke({"context": doc_str, "question": question})
    return {"generation": generation}

def generate_fallback_answer_node(state):
    """Generates an answer using the LLM's general knowledge."""
    print("---NODE: GENERATE FALLBACK ANSWER---")
    question = state["condensed_question"]
    
    fallback_chain = llm | StrOutputParser()
    generation = fallback_chain.invoke(question)
    return {"generation": generation}

# --- Build and Compile the Graph ---
workflow = StateGraph(GraphState)

# Get the router node from our new agent file
tool_router = get_tool_router_node(llm)

# Add all nodes to the graph
workflow.add_node("condense_question", condense_question_node)
workflow.add_node("router", tool_router)
workflow.add_node("retrieve", retrieve_documents_node)
workflow.add_node("run_wikipedia", lambda state: run_wikipedia_node(state, llm))
workflow.add_node("grade", grade_documents_node)
workflow.add_node("generate_rag", generate_rag_answer_node)
workflow.add_node("generate_fallback", generate_fallback_answer_node)

# Define the graph's flow
workflow.set_entry_point("condense_question")
workflow.add_edge("condense_question", "router")

# The router decides the next step
workflow.add_conditional_edges("router", lambda x: x["source"], {
    "wikipedia": "run_wikipedia",
    "vectorstore": "retrieve"
})

# The original flow for retrieving and grading documents
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", lambda x: x["decision"], {
    "generate_rag": "generate_rag",
    "generate_fallback": "generate_fallback",
})

# After running wikipedia, we generate an answer from its output
workflow.add_edge("run_wikipedia", "generate_rag")

# Define the end points of the graph
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
            documents=chunks, 
            embedding=embeddings, 
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY, 
            collection_name=QDRANT_COLLECTION_NAME, 
            force_recreate=False # Set to False to add to existing collection
        )

        return {"status": "success", "message": f"PDF '{file.filename}' processed and added to the knowledge base."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        await file.close()

@app.post("/ask-question")
async def ask_question(request: Question):
    try:
        inputs = {"question": request.query, "chat_history": request.chat_history}
        final_state = app_graph.invoke(inputs)
        return {"answer": final_state["generation"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")