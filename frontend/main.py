import streamlit as st
import requests

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000"

# --- Page Setup ---
st.set_page_config(page_title="DocuMentor", page_icon="📚", layout="wide")
st.title("📚 DocuMentor: Your AI Study Assistant")
st.write("Upload a PDF to start a chat, or use one of the agents to kickstart your learning!")

# --- Session State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "study_plan" not in st.session_state: st.session_state.study_plan = None
if "quiz_data" not in st.session_state: st.session_state.quiz_data = None
if "quiz_submitted" not in st.session_state: st.session_state.quiz_submitted = False
if "user_answers" not in st.session_state: st.session_state.user_answers = {}

# --- Helper Functions ---
def fetch_skill(category):
    # ... (fetch_skill function remains the same as before) ...
    try:
        with st.spinner(f"🤖 Generating a {category} skill for you..."):
            response = requests.get(f"{BACKEND_URL}/get-skill/{category}")
        if response.status_code == 200:
            skill_data = response.json()
            st.session_state.messages.append({"role": "user", "content": f"Give me a {category} skill."})
            content = (f"**Here's your {category} tip from the '{skill_data['category_type']}' category:**\n\n### {skill_data['title']}\n\n{skill_data['content']}")
            st.session_state.messages.append({"role": "assistant", "content": content})
        else: st.error(f"Could not retrieve skill: {response.text}")
    except requests.exceptions.RequestException: st.error("Connection Error")

# --- Sidebar ---
with st.sidebar:
    st.header("RAG Chatbot")
    pdf_processed = "pdf_processed" in st.session_state and st.session_state.pdf_processed
    uploaded_file = st.file_uploader("Upload a PDF to ask questions", type="pdf")
    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Embedding your document..."):
                # ... (PDF processing logic is the same)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload-pdf", files=files)
                    if response.status_code == 200:
                        st.session_state.messages = [{"role": "assistant", "content": f"Hi! I'm ready to answer questions about `{uploaded_file.name}`."}]
                        st.session_state.study_plan = None
                        st.session_state.pdf_processed = True # Flag that a PDF is ready
                        st.success("PDF processed!")
                    else: st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.RequestException: st.error("Connection Error")
    st.divider()

    st.header("Planner Agent")
    # ... (Planner Agent UI is the same)
    with st.expander("📅 Create a Revision Schedule"):
        days = st.number_input("Days to plan?", min_value=1, max_value=90, value=7, key="p_d")
        subjects = st.text_area("Subjects to cover?", "DSA, CN", key="p_s")
        if st.button("Generate Plan", key="p_b"):
            if subjects:
                # ... (Planner logic is the same)
                with st.spinner("🤖 Planner is creating your schedule..."):
                    payload = {"days": days, "subjects": subjects}
                    try:
                        response = requests.post(f"{BACKEND_URL}/create-plan", json=payload)
                        if response.status_code == 200:
                            st.session_state.study_plan = response.json().get('plan')
                            st.session_state.messages, st.session_state.quiz_data = [], None
                        else: st.error(f"Error: {response.json().get('detail')}")
                    except requests.exceptions.RequestException: st.error("Connection Error")
            else: st.warning("Please enter subjects.")
    st.divider()

    # --- NEW QUIZ AGENT UI ---
    st.header("Quiz Agent")
    with st.expander("📝 Take a Quiz"):
        topic = st.text_input("What topic do you want a quiz on?", "B+ Trees", key="quiz_topic")
        use_pdf = st.checkbox("Use uploaded PDF as context", value=True, disabled=not pdf_processed)
        if not pdf_processed: st.caption("Upload and process a PDF to enable context-based quizzes.")
        
        if st.button("Generate Quiz", key="quiz_btn"):
            if topic:
                with st.spinner("🧠 The Quiz Agent is generating questions..."):
                    payload = {"topic": topic, "use_pdf": use_pdf}
                    try:
                        response = requests.post(f"{BACKEND_URL}/create-quiz", json=payload)
                        if response.status_code == 200:
                            st.session_state.quiz_data = response.json()
                            st.session_state.quiz_submitted = False
                            st.session_state.user_answers = {}
                            st.session_state.messages, st.session_state.study_plan = [], None
                        else: st.error(f"Error: {response.json().get('detail')}")
                    except requests.exceptions.RequestException: st.error("Connection Error")
            else: st.warning("Please enter a topic for the quiz.")

# --- Main Content Area ---
# The main area now has three modes: Plan, Quiz, or Chat
if st.session_state.study_plan:
    # ... (Study plan display logic is the same)
    st.header("Your Custom Study Plan")
    st.markdown(st.session_state.study_plan)
    if st.button("Clear Plan and Start Chat"): st.session_state.study_plan = None; st.rerun()
elif st.session_state.quiz_data:
    st.header(f"Quiz on: {st.session_state.quiz_data.get('topic', 'your topic')}")
    if not st.session_state.quiz_submitted:
        with st.form("quiz_form"):
            for i, q in enumerate(st.session_state.quiz_data["questions"]):
                st.subheader(f"Question {i+1}: {q['question_text']}")
                st.session_state.user_answers[i] = st.radio("Choose one:", q['options'], key=f"q_{i}")
            
            if st.form_submit_button("Submit Answers"):
                st.session_state.quiz_submitted = True
                st.rerun()
    else: # If quiz is submitted, show score
        score = 0
        for i, q in enumerate(st.session_state.quiz_data["questions"]):
            if st.session_state.user_answers[i] == q['correct_answer']:
                score += 1
        
        st.success(f"### Your Score: {score} / {len(st.session_state.quiz_data['questions'])}")
        st.balloons()
        
        with st.expander("Review Answers"):
            for i, q in enumerate(st.session_state.quiz_data["questions"]):
                user_ans = st.session_state.user_answers[i]
                correct_ans = q['correct_answer']
                if user_ans == correct_ans:
                    st.write(f"**Question {i+1}: {q['question_text']}**")
                    st.write(f"✔️ Your answer: **{user_ans}** (Correct!)")
                else:
                    st.write(f"**Question {i+1}: {q['question_text']}**")
                    st.write(f"❌ Your answer: **{user_ans}** (Incorrect)")
                    st.write(f"➡️ Correct answer: **{correct_ans}**")
        
        if st.button("Take Another Quiz"):
            st.session_state.quiz_data = None
            st.rerun()
else: # Default view is the chat/skill interface
    # ... (Chat and Skill button logic is the same)
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])
    if not st.session_state.messages:
        st.info("Start a conversation by getting a skill, uploading a PDF, or using an agent from the sidebar!")
        cols = st.columns(3);
        if cols[0].button("🧠 Get Tech Skill", use_container_width=True): fetch_skill("Tech"); st.rerun()
        if cols[1].button("📈 Get Aptitude Skill", use_container_width=True): fetch_skill("Aptitude"); st.rerun()
        if cols[2].button("🤝 Get Soft Skill", use_container_width=True): fetch_skill("Soft Skills"); st.rerun()
    if prompt := st.chat_input("What would you like to know?"):
        # ... (Chat submission logic is the same)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        chat_history = []
        for i, msg in enumerate(st.session_state.messages[:-1]):
            if msg["role"] == "user":
                if i + 1 < len(st.session_state.messages) and st.session_state.messages[i+1]["role"] == "assistant":
                    chat_history.append((msg["content"], st.session_state.messages[i+1]["content"]))
        payload = {"query": prompt, "chat_history": chat_history}
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/ask-question", json=payload)
                    if response.status_code == 200:
                        answer = response.json().get("answer")
                        st.markdown(answer); st.session_state.messages.append({"role": "assistant", "content": answer})
                    else: st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.RequestException: st.error("Connection Error")