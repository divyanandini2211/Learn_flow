#  Your AI Study Assistant

A multi-agent AI application that blends deep document analysis with practical career skills, designed to be the ultimate study partner for college students. Powered by LangGraph, Google Gemini, and Qdrant.

---

## ✨ Project Vision

Standard study tools are often one-dimensional. DocuMentor is built on the idea that students need a holistic assistant that helps with both **academic revision** and **career-focused micro-learning**. This application combines a powerful RAG (Retrieval-Augmented Generation) chatbot for course notes with a suite of standalone AI agents for other essential tasks.

## 🚀 Core Features

*   **🧠 Conversational RAG Agent:** Upload any PDF (lecture notes, research papers, textbooks) and have a deep, context-aware conversation. The agent remembers your chat history and uses a sophisticated LangGraph-powered workflow to decide whether to use the PDF's knowledge or its own.
*   **💡 Dynamic Skill Agent:** Get a unique, AI-generated micro-skill on demand. The agent is trained to provide relevant, college-level tips across three categories: Tech, Aptitude, and Soft Skills.
*   **📅 AI-Powered Study Planner:** Input your study duration and subjects, and the Planner Agent will generate a detailed, day-by-day revision schedule in a clean markdown table.
*   **📝 Interactive Quiz Agent:** Generate a 5-question multiple-choice quiz on any topic, either from a general knowledge base or based specifically on the content of your uploaded PDF.

## 🛠️ Tech Stack & Architecture

This project uses a modern, modular architecture with a separate backend and frontend.

*   **Backend:** FastAPI, LangGraph, LangChain
*   **Frontend:** Streamlit
*   **AI & ML:** Google Gemini 1.5 Flash, Google Embeddings
*   **Database:** Qdrant Cloud (Vector Store)

## ⚙️ Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

*   Python 3.9+
*   Git
*   A [Qdrant Cloud](https://cloud.qdrant.io/) account
*   A [Google AI Studio](https://aistudio.google.com/) API Key

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Backend Setup:**
    *   Navigate to the backend directory: `cd backend`
    *   Create a `.env` file and add your keys (no quotes):
        ```env
        GOOGLE_API_KEY=your_google_api_key
        QDRANT_URL=your_qdrant_cloud_url
        QDRANT_API_KEY=your_qdrant_api_key
        ```
    *   Create and activate a Python virtual environment:
        ```bash
        python -m venv venv
        # Windows: venv\Scripts\activate
        # macOS/Linux: source venv/bin/activate
        ```
    *   Install libraries: `pip install -r requirements.txt`

3.  **Frontend Setup:**
    *   Navigate to the frontend directory: `cd ../frontend`
    *   Create and activate a separate virtual environment.
    *   Install libraries: `pip install -r requirements.txt`

### Running the Application

1.  **Start the Backend (in Terminal 1, from the `backend` folder):**
    ```bash
    uvicorn backend:app --reload
    ```
2.  **Start the Frontend (in Terminal 2, from the `frontend` folder):**
    ```bash
    streamlit run main.py
    ```
    Open your browser to `http://localhost:8501`.