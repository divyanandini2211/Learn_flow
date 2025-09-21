import os
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables for this file
load_dotenv()

# Create a new, separate router for this agent
router = APIRouter()

# Initialize the LLM specifically for this agent's tasks
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)  # Lower temperature for more consistent formatting

# --- Pydantic Model for incoming plan requests ---
class PlanRequest(BaseModel):
    days: int
    subjects: str

def clean_and_format_plan(raw_plan: str) -> str:
    """
    Clean the AI-generated plan and ensure proper markdown table formatting.
    """
    # Remove all HTML tags
    clean_text = re.sub(r'<[^>]+>', '', raw_plan)
    
    # Remove extra whitespace and normalize line breaks
    clean_text = re.sub(r'\n\s*\n', '\n', clean_text)
    clean_text = clean_text.strip()
    
    # If the output doesn't contain a proper markdown table, try to extract and reformat
    if '|' not in clean_text or 'Day' not in clean_text:
        # Fallback: create a simple table structure
        lines = clean_text.split('\n')
        table_lines = []
        table_lines.append("| Day | Subject(s) | Topics to Cover / Goals |")
        table_lines.append("|-----|------------|-------------------------|")
        
        current_day = 1
        current_subject = ""
        current_topics = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(('Day', 'day')):
                if current_topics:
                    topics_str = "; ".join(current_topics)
                    table_lines.append(f"| Day {current_day} | {current_subject} | {topics_str} |")
                    current_topics = []
                    current_day += 1
                # Extract subject from the day line if present
                parts = line.split(':')
                if len(parts) > 1:
                    current_subject = parts[1].strip()
            elif line.startswith(('•', '-', '*')):
                current_topics.append(line.lstrip('•-* '))
            elif current_subject == "" and not line.startswith(('Day', 'day')):
                current_subject = line
            else:
                current_topics.append(line)
        
        # Add the last entry
        if current_topics:
            topics_str = "; ".join(current_topics)
            table_lines.append(f"| Day {current_day} | {current_subject} | {topics_str} |")
        
        return '\n'.join(table_lines)
    
    return clean_text

# --- Define the Planner Agent Endpoint ON THE ROUTER ---
@router.post("/create-plan")
async def create_plan(request: PlanRequest):
    """
    Generates a personalized study plan using the Gemini LLM with guaranteed clean table formatting.
    """
    try:
        # Enhanced prompt with strict formatting requirements
        prompt = PromptTemplate.from_template(
            """You are an expert study planner. Create a {days}-day revision schedule for the following subjects: {subjects}

IMPORTANT FORMATTING REQUIREMENTS:
- Output MUST be a clean markdown table format
- Use ONLY plain text and markdown table syntax (|, -, etc.)
- Do NOT use any HTML tags like <br>, <div>, <p>, etc.
- Do NOT use bullet points or numbered lists outside the table cells
- Keep topic descriptions concise and clear

Create a table with exactly these three columns:
| Day | Subject(s) | Topics to Cover / Goals |
|-----|------------|-------------------------|

For each day, specify:
- Day number (Day 1, Day 2, etc.)
- Subject name(s) for that day
- Specific topics, concepts, or goals (use semicolons to separate multiple items)

Example format:
| Day 1 | Operating Systems | Define OS functions; Understand process states; Learn about PCBs |
| Day 2 | Operating Systems | Process scheduling algorithms; Context switching; Thread concepts |

Generate the complete {days}-day schedule now:"""
        )
        
        planner_chain = prompt | llm | StrOutputParser()
        
        # Generate the plan
        raw_plan = planner_chain.invoke({
            "days": request.days,
            "subjects": request.subjects
        })
        
        # Clean and format the plan
        clean_plan = clean_and_format_plan(raw_plan)
        
        return {"plan": clean_plan}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate plan: {e}")

# Additional endpoint to reformat existing messy plans
@router.post("/clean-plan")
async def clean_existing_plan(plan_text: str):
    """
    Clean and reformat an existing messy study plan.
    """
    try:
        clean_plan = clean_and_format_plan(plan_text)
        return {"plan": clean_plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clean plan: {e}")