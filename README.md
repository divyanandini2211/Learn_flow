# Learn Flow – Your AI Study Assistant

A multi-agent AI application that blends deep document analysis with practical career skills, designed to be the ultimate study partner for college students. Powered by LangGraph, Google Gemini, and Qdrant, Learn Flow is your companion for smarter revision, skill-building, and academic success.

<img width="1880" height="868" alt="image" src="https://github.com/user-attachments/assets/6f1ec876-b5d5-42db-b5b8-3de7731d842c" />

![Learn Flow UI](<img width="1880" height="868" alt="image" src="https://github.com/user-attachments/assets/6f1ec876-b5d5-42db-b5b8-3de7731d842c" />)


## Project Vision

Traditional study tools are often one-dimensional, focusing only on content recall.
Learn Flow is built on the belief that students need a holistic AI assistant that supports both:

*   **Academic Learning** – Through document-based RAG-powered conversations
*   **Career Readiness** – With AI-generated micro-skills and practice tools

By combining course revision, study planning, skill development, and interactive quizzes, Learn Flow transforms the way students prepare for exams and placements.

## Core Features

### Conversational RAG Agent
*   Upload any PDF (lecture notes, research papers, textbooks).
*   Ask deep, context-aware questions and receive precise answers.

<!-- 
    ↓↓↓ 2. SCREENSHOT FOR PDF CHAT GOES HERE ↓↓↓ 
-->
![Screenshot of PDF Chat](<img width="1842" height="737" alt="image" src="https://github.com/user-attachments/assets/b481458d-66e4-491b-a73b-527ea79430ae" />
)


### Wikipedia Search Tool
*   Ask factual questions using keywords like "wiki" to get comprehensive summaries directly from Wikipedia, complete with source citations.

<!-- 
    ↓↓↓ 3. SCREENSHOT FOR WIKIPEDIA SEARCH GOES HERE ↓↓↓ 
-->
![Screenshot of Wikipedia Search](https://via.placeholder.com/600x350.png?text=Screenshot+of+Wikipedia+Search+Here)


### Dynamic Skill Agent
*   Get AI-generated micro-skills on demand.
*   Three categories: Tech Skills, Aptitude, and Soft Skills.

<!-- 
    ↓↓↓ 4. SCREENSHOT FOR SKILL AGENT GOES HERE ↓↓↓ 
-->
![Screenshot of Skill Agent](<img width="1517" height="718" alt="image" src="https://github.com/user-attachments/assets/46a32793-a886-4b75-9dd9-528905d4818b" />
)


### AI-Powered Study Planner
*   Input your study duration & subjects.
*   Generate a day-by-day revision plan in a clean markdown table.

<!-- 
    ↓↓↓ 5. SCREENSHOT FOR STUDY PLANNER GOES HERE ↓↓↓ 
-->
![Screenshot of Study Planner](<img width="1836" height="799" alt="image" src="https://github.com/user-attachments/assets/5f4ad1bb-7fef-4129-bd29-b22daef60ebe" />
)


### Interactive Quiz Agent
*   Instantly create 5-question MCQs on any topic.
*   Quiz can be based on general knowledge or your uploaded PDF content.

<!-- 
    ↓↓↓ 6. SCREENSHOT FOR QUIZ AGENT GOES HERE ↓↓↓ 
-->
![Screenshot of Quiz Agent](<img width="1785" height="881" alt="image" src="https://github.com/user-attachments/assets/b1edc75f-dec2-4232-9340-a971157351a4" />
)


## Tech Stack & Architecture

*   **Backend**: FastAPI (API framework), LangGraph + LangChain (multi-agent orchestration)
*   **Frontend**: Streamlit (lightweight, interactive UI)
*   **AI & ML**: Google Gemini, Google Embeddings
*   **Database**: Qdrant Cloud (vector store for PDF context retrieval)

## 📂 Project Structure
```
Learn_flow/
├── backend/                 # FastAPI + LangGraph backend
│   ├── backend.py           # Main FastAPI application
│   ├── agents/              # AI agents (RAG, Quiz, Planner, Skills)
│   └── requirements.txt     # Backend dependencies
├── frontend/                # Streamlit frontend
│   ├── main.py              # UI entry point
│   └── requirements.txt     # Frontend dependencies
├── .env.example             # Example environment variables
└── README.md                # Project documentation
```

## Getting Started
### Prerequisites

*   Python 3.9+
*   Git
*   A [Qdrant Cloud](https://cloud.qdrant.io/) account
*   A [Google AI Studio](https://aistudio.google.com/) API key

### Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/divyanandini2211/Learn_flow.git
    cd Learn_flow
    ```

2.  **Backend Setup**
    ```bash
    cd backend
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    pip install -r requirements.txt
    ```

    Create a `.env` file inside `backend/` and add:
    ```
    GOOGLE_API_KEY=your_google_api_key
    QDRANT_URL=your_qdrant_cloud_url
    QDRANT_API_KEY=your_qdrant_api_key
    ```

3.  **Frontend Setup**
    ```bash
    cd ../frontend
    python -m venv venv
    source venv/bin/activate   # (or venv\Scripts\activate on Windows)
    pip install -r requirements.txt
    ```

## ▶ Running the Application

1.  **Start Backend** (Terminal 1 – inside `backend/`):
    ```bash
    uvicorn backend:app --reload
    ```

2.  **Start Frontend** (Terminal 2 – inside `frontend/`):
    ```bash
    streamlit run main.py
    ```

### Access in Browser:

*   **Frontend**: `http://localhost:8501`
*   **Backend API**: `http://localhost:8000`

## Usage Examples

### Academic Assistance
*   Upload lecture notes → “Explain this theorem step by step.”
*   “Summarize Chapter 3 in 5 key points.”

### Skill Development
*   “Give me an aptitude tip.”

### Study Planning
*   “Plan a 7-day schedule for OS and DSA.”

### Interactive Quiz
*   “Generate 5 MCQs from this PDF.”
*   “Quiz me on probability.”

## Design Philosophy

*   **Student-Centric**: Tailored for college students preparing for exams & placements.
*   **Multi-Agent Intelligence**: Each agent specializes in a unique task.
*   **Simplicity First**: Minimalist frontend for fast, distraction-free usage.
*   **Scalability**: Modular architecture for adding more agents in the future.

## Future Enhancements

- [ ] **Progress Tracker** – Monitor study progress & weak areas
- [ ] **Mobile App** – Cross-platform availability
- [ ] **Collaboration** – Study groups & shared plans

## Contributors

*   Divya Nandini- 23N217
*   Pranika - 23N236
*   Thejashri  - 23N262
*   Priyanka - 24N434

## License

This project is developed as part of an academic assignment.
All rights reserved.

---
**Learn Flow – Smarter study, stronger future.**
