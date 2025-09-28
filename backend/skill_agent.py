import os
import random # <--- IMPORT THE RANDOM LIBRARY
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
router = APIRouter()

# The cache=False flag is good practice, so we'll keep it.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.9, cache=False)

# Pydantic Model for the response
class SkillResponse(BaseModel):
    title: str = Field(description="The main title of the skill, concept, or question.")
    content: str = Field(description="The main body of the response, explaining the concept or providing the question's method and answer.")
    category_type: str = Field(description="A more specific sub-category, e.g., 'Data Structures', 'Logical Reasoning', 'Interview Prep'.")

@router.get("/get-skill/{category}")
async def get_skill_by_category(category: str, response: Response):
    """
    Generates a unique skill by adding a random seed to the prompt,
    making it impossible for any cache to return a stale result.
    """
    if category == "Tech":
        prompt_text = """You are a computer science professor. Generate one single, useful micro-skill or concept for a college student.
        The topic should be from a core CS subject like Databases (DBMS), Computer Networks (CN), Operating Systems (OS), OOPS, or Data Structures and Algorithms (DSA).
        
        To ensure a unique response, use this random seed as inspiration: {random_seed}.
        {format_instructions}"""
    elif category == "Aptitude":
        prompt_text = """You are an aptitude test expert coaching a college student for placements.
        Generate one single aptitude question (quantitative or logical).
        In the 'content' field, you MUST provide the Question, the final Answer, and a short explanation of the Method to solve it.
        
        To ensure a unique response, use this random seed as inspiration: {random_seed}.
        {format_instructions}"""
    elif category == "Soft Skills":
        prompt_text = """You are a career coach advising a college student.
        Generate one single, actionable soft skill tip relevant for internships or job interviews.
        
        To ensure a unique response, use this random seed as inspiration: {random_seed}.
        {format_instructions}"""
    else:
        raise HTTPException(status_code=404, detail="Skill category not found")

    try:
        # We add "random_seed" to the input variables for the prompt
        prompt = PromptTemplate.from_template(
            template=prompt_text,
            partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=SkillResponse).get_format_instructions()},
        )
        
        parser = PydanticOutputParser(pydantic_object=SkillResponse)
        chain = prompt | llm | parser
        
        # --- THE GUARANTEED FIX ---
        # We invoke the chain with a new random number every single time.
        # This makes the input unique, forcing a new generation.
        skill_object = chain.invoke({
            "random_seed": random.randint(0, 100000)
        })
        
        # Keep the browser cache headers as a backup safety measure
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        return skill_object.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate skill: {e}")