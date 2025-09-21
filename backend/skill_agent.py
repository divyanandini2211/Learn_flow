import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
router = APIRouter()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8)

# --- Pydantic Models for Structured Output ---
# We can use one flexible model for all categories now
class SkillResponse(BaseModel):
    title: str = Field(description="The main title of the skill, concept, or question.")
    content: str = Field(description="The main body of the response, explaining the concept or providing the question's method and answer.")
    category_type: str = Field(description="A more specific sub-category, e.g., 'Data Structures', 'Logical Reasoning', 'Interview Prep'.")

# --- Define the Skill Agent Endpoint ---
@router.get("/get-skill/{category}")
async def get_skill_by_category(category: str):
    """
    Generates a unique, college-student-focused skill based on the chosen high-level category.
    """
    if category == "Tech":
        prompt = PromptTemplate.from_template(
            """You are a computer science professor. Generate one single, useful micro-skill or concept for a college student.
            The topic should be from a core CS subject like Databases (DBMS), Computer Networks (CN), Operating Systems (OS), OOPS, or Data Structures and Algorithms (DSA).
            
            {format_instructions}"""
        )
    elif category == "Aptitude":
        prompt = PromptTemplate.from_template(
            """You are an aptitude test expert coaching a college student for placements.
            Generate one single aptitude question (quantitative or logical).
            In the 'content' field, you MUST provide the Question, the final Answer, and a short explanation of the Method to solve it.
            
            {format_instructions}"""
        )
    elif category == "Soft Skills":
        prompt = PromptTemplate.from_template(
            """You are a career coach advising a college student.
            Generate one single, actionable soft skill tip relevant for internships or job interviews.
            
            {format_instructions}"""
        )
    else:
        raise HTTPException(status_code=404, detail="Skill category not found")

    try:
        parser = PydanticOutputParser(pydantic_object=SkillResponse)
        chain = prompt | llm | parser
        skill_object = chain.invoke({"format_instructions": parser.get_format_instructions()})
        
        return skill_object.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate skill: {e}")