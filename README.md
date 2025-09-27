 DocuMentor – Your AI Study Assistant

A multi-agent AI application that blends deep document analysis with practical career skills, designed to be the ultimate study partner for college students. Powered by LangGraph, Google Gemini, and Qdrant, DocuMentor is your companion for smarter revision, skill-building, and academic success.

 Project Vision

Traditional study tools are often one-dimensional, focusing only on content recall.
DocuMentor is built on the belief that students need a holistic AI assistant that supports both:

Academic Learning – Through document-based RAG-powered conversations

Career Readiness – With AI-generated micro-skills and practice tools

By combining course revision, study planning, skill development, and interactive quizzes, DocuMentor transforms the way students prepare for exams and placements.

 Core Features
 Conversational RAG Agent

Upload any PDF (lecture notes, research papers, textbooks).

Ask deep, context-aware questions and receive precise answers.

Conversation memory ensures continuity across sessions.

Powered by a LangGraph workflow to decide when to use PDF knowledge vs general AI knowledge.

 Dynamic Skill Agent

Get AI-generated micro-skills on demand.

Three categories: Tech Skills, Aptitude, and Soft Skills.

Perfect for career prep and building well-rounded knowledge.

 AI-Powered Study Planner

Input your study duration & subjects.

Generate a day-by-day revision plan in a clean markdown table.

Tailored to your schedule and priorities.

 Interactive Quiz Agent

Instantly create 5-question MCQs on any topic.

Quiz can be based on general knowledge or your uploaded PDF content.

Great for self-testing and memory retention.

 Tech Stack & Architecture

Backend:

FastAPI (API framework)

LangGraph + LangChain (multi-agent orchestration)

Frontend:

Streamlit (lightweight, interactive UI)

AI & ML:

Google Gemini 1.5 Flash

Google Embeddings

Database:

Qdrant Cloud (vector store for PDF context retrieval)

📂 Project Structure
DocuMentor/
├── backend/                 # FastAPI + LangGraph backend
│   ├── backend.py           # Main FastAPI application
│   ├── agents/              # AI agents (RAG, Quiz, Planner, Skills)
│   └── requirements.txt     # Backend dependencies
├── frontend/                # Streamlit frontend
│   ├── main.py              # UI entry point
│   └── requirements.txt     # Frontend dependencies
├── .env.example             # Example environment variables
└── README.md                # Project documentation

 Getting Started
 Prerequisites

Python 3.9+

Git

A Qdrant Cloud
 account

A Google AI Studio
 API key

 Installation & Setup

Clone the repository

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Backend Setup

cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt


Create a .env file inside backend/ and add:

GOOGLE_API_KEY=your_google_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key


Frontend Setup

cd ../frontend
python -m venv venv
source venv/bin/activate   # (or venv\Scripts\activate on Windows)
pip install -r requirements.txt

 Running the Application

Start Backend (Terminal 1 – inside backend/):

uvicorn backend:app --reload


Start Frontend (Terminal 2 – inside frontend/):

streamlit run main.py


Access in Browser:

Frontend: http://localhost:8501

Backend API: http://localhost:8000

 Usage Examples

Academic Assistance

Upload lecture notes → “Explain this theorem step by step.”

“Summarize Chapter 3 in 5 key points.”

Skill Development

“Give me an aptitude tip.”

“Suggest a soft skill for interviews.”

Study Planning

“Plan a 7-day schedule for OS and DSA.”

Interactive Quiz

“Generate 5 MCQs from this PDF.”

“Quiz me on probability.”

 Design Philosophy

Student-Centric: Tailored for college students preparing for exams & placements.

Multi-Agent Intelligence: Each agent specializes in a unique task.

Simplicity First: Minimalist frontend for fast, distraction-free usage.

Scalability: Modular architecture for adding more agents in the future.

 Future Enhancements

 Progress Tracker – Monitor study progress & weak areas

 Placement Prep Toolkit – Resume feedback, mock interviews, GD practice

 Video Integration – Explanations & micro-lectures

 Mobile App – Cross-platform availability

 Collaboration – Study groups & shared plans

 Contributors

Your Name Here – Project Owner & Developer

 License

This project is developed as part of an academic assignment.
All rights reserved.

 DocuMentor – Smarter study, stronger future.
