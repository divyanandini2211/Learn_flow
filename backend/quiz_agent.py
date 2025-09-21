import os
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Qdrant and LangChain imports for retrieval
import qdrant_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
router = APIRouter()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5) # Lower temp for more factual quizzes

# --- Pydantic Models for a perfectly structured quiz ---
class QuizQuestion(BaseModel):
    question_text: str = Field(description="The text of the multiple-choice question.")
    options: List[str] = Field(description="A list of 4 possible answers (A, B, C, D).")
    correct_answer: str = Field(description="The correct answer from the provided options list.")

class Quiz(BaseModel):
    questions: List[QuizQuestion] = Field(description="A list of 5 multiple-choice questions.")

class QuizRequest(BaseModel):
    topic: str
    use_pdf: bool

# --- Define the Quiz Agent Endpoint ---
@router.post("/create-quiz")
async def create_quiz(request: QuizRequest):
    """
    Generates a 5-question multiple-choice quiz.
    Conditionally uses context from the uploaded PDF if requested.
    """
    try:
        context = ""
        # --- Conditional Logic: Use PDF or General Knowledge ---
        if request.use_pdf:
            print("---QUIZ AGENT: Retrieving context from PDF---")
            # Re-initialize the retriever to access the PDF's knowledge base
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            client = qdrant_client.QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
            vector_store = Qdrant(client=client, collection_name="student_notes_collection", embeddings=embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # Get more context for a quiz
            
            # Retrieve documents and format them as context
            retrieved_docs = retriever.invoke(request.topic)
            if not retrieved_docs:
                raise HTTPException(status_code=404, detail=f"Could not find any context for the topic '{request.topic}' in the PDF.")
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            prompt_template = """Based ONLY on the following context, generate a 5-question multiple-choice quiz about {topic}.
            
            Context:
            {context}
            
            {format_instructions}
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["topic", "context"], partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=Quiz).get_format_instructions()})
            input_vars = {"topic": request.topic, "context": context}
        else:
            print("---QUIZ AGENT: Using general knowledge---")
            prompt_template = "You are a teacher. Generate a 5-question multiple-choice quiz about {topic} for a college student.\n\n{format_instructions}"
            prompt = PromptTemplate(template=prompt_template, input_variables=["topic"], partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=Quiz).get_format_instructions()})
            input_vars = {"topic": request.topic}

        parser = PydanticOutputParser(pydantic_object=Quiz)
        chain = prompt | llm | parser
        quiz_object = chain.invoke(input_vars)
        
        return quiz_object.dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate quiz: {e}")