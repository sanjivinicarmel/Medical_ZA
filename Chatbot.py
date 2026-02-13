import os
#from requests import session
import streamlit as st
from dotenv import load_dotenv
import json
import uuid
import os

TRIAGE_DIR = "triage_sessions"
os.makedirs(TRIAGE_DIR, exist_ok=True)

load_dotenv(".env")

st.set_page_config(page_icon="💊", page_title="Medical Assistant", layout="wide")

# ---------------------------
# System Prompt
# ---------------------------
SYSTEM_PROMPT = (
    "You are a friendly, helpful medical chat assistant.\n"
    "- Use simple language\n"
    "- Never diagnose\n"
    "- Give helpful suggestions\n"
    "- Ask one gentle follow-up question\n"
    "- Recommend doctor only if severe/persistent\n"
    "- End with: 'This is general information and not a substitute for professional medical advice.'"
)

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]

if "user_word_count" not in st.session_state:
    st.session_state.user_word_count = 0

if "show_intro" not in st.session_state:
    st.session_state.show_intro = True

# IMPROVED FULL-WIDTH STYLING
st.markdown("""
<style>
/* Force full width container - remove all padding constraints */
.main .block-container{
    max-width: 100% !important;
    padding-top: 0.5rem !important;
    padding-left: 0.5rem !important;
    padding-right: 0.5rem !important;
    padding-bottom: 2rem !important;
}

/* Full mint background */
body, .stApp {
    background-color: #a5e6d5;
}

/* Remove wrapper constraints - let content stretch */
.landing-wrapper{
    width: 100%;
    max-width: 100%;
    margin: 0;
    padding: 0 0.5rem;
}

/* NAVY HEADER - Full width */
.top-box{
    background: #032B63;
    color: white;
    padding: 50px 60px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 25px;
    width: 100%;
}

/* GREEN BOX - Full width */
.green-box{
    background: #6FB449;
    color: white;
    padding: 45px 60px;
    border-radius: 15px;
    margin-bottom: 25px;
    width: 100%;
}

/* BLUE BOX - Full width */
.blue-box{
    background: #0073B4;
    color: white;
    padding: 45px 60px;
    border-radius: 15px;
    margin-bottom: 40px;
    width: 100%;
}

/* Divider dashed */
.divider{
    width: 100%;
    border-top: 4px dashed #333;
    margin: 25px 0;
}

.big-title{
    font-size: 38px;
    font-weight: 900;
    margin-bottom: 15px;
}

.triage-header {
    font-size: 28px;
    font-weight: 800;
    margin-top: 40px;
    margin-bottom: 12px;
    color: #1f2937;
}

.triage-info-box {
    background: #e9f8f4;
    border-left: 6px solid #0073B4;
    padding: 20px 24px;
    border-radius: 14px;
    margin-bottom: 24px;
    line-height: 1.6;
    font-size: 17px;
}
                        
            
/* AI response card */
.ai-response-box {
    background: #e9f8f4;
    border-left: 6px solid #0073B4;
    padding: 18px 22px;
    border-radius: 12px;
    margin: 16px 0;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* User message card (optional, lighter) */
.user-message-box {
    background: #dff3ee;
    padding: 14px 18px;
    border-radius: 10px;
    margin: 12px 0;
}


.sub-title{
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 15px;
}

.text{
    font-size: 18px;
    line-height: 1.7;
}

/* Chat section styling - Full width */
.consultation-section {
    width: 100%;
    max-width: 100%;
    margin: 0;
    padding: 0 0.5rem;
}

/* Ensure Streamlit elements respect the width */
.stChatInput {
    max-width: 100% !important;
}

.stButton button {
    width: auto;
    padding: 0.5rem 2rem;
}

/* Override any Streamlit default margins */
.element-container {
    width: 100% !important;
}
            
/* TRIAGE HEADER (Dark Blue) */
.triage-title-box{
    background:#032B63;
    color:white;
    padding:28px 32px;
    border-radius:18px;
    font-size:26px;
    font-weight:800;
    display:flex;
    align-items:center;
    gap:12px;
    margin-top:40px;
}

/* TRIAGE DESCRIPTION BOX (Blue) */
.triage-desc-box{
    background:#0073B4;
    color:white;
    padding:28px 32px;
    border-radius:18px;
    font-size:17px;
    line-height:1.7;
    margin-top:18px;
}

/* Make text bold highlights readable */
.triage-desc-box b{
    font-weight:700;
}

/* Chat input wrapper */
.chat-section {
    background: #aef0df;
    padding: 30px;
    border-radius: 18px;
    margin-top: 30px;
}

/* Chat input label */
.chat-title {
    font-size: 26px;
    font-weight: 800;
    margin-bottom: 8px;
    color: #1f2d3d;
}

/* Chat subtitle */
.chat-subtitle {
    font-size: 15px;
    margin-bottom: 16px;
    color: #2c3e50;
}

/* Slight spacing fix */
section[data-testid="stChatInput"] {
    margin-top: 0px;
}            
</style>
""", unsafe_allow_html=True)


# ---------------- LANDING CONTENT ----------------
st.markdown("""
<div class="landing-wrapper">

<div class="top-box">
<h1 class="big-title">🩺 AI Symptom Intake</h1>
<p class="sub-title">
From First Symptoms to Structured Clinical Insights — Fast, Accurate, and Context-Aware.
</p>
<p class="text">
Streamline assessment with AI-powered symptom capture, automated triage reasoning,
and seamless integration into the patient journey.
</p>
</div>

<div class="green-box">
<p class="text">
This application enables patients to describe their symptoms through an interactive AI chatbot.
As the conversation progresses, the system captures key clinical details, interprets the patient's inputs,
and intelligently analyzes the symptom pattern. Using medical reasoning, the chatbot performs a preliminary
disease shortlisting, offering an early indication of possible conditions before formal diagnosis.
</p>
</div>

<div class="divider"></div>

<div class="blue-box">
<p class="text">
This section allows patients to describe their symptoms naturally through an AI-powered chatbot.
The assistant engages in a guided conversation, asks clarifying questions, and captures clinically
relevant details in real time. By analyzing the dialogue, the system identifies key symptom patterns
and performs an initial shortlisting of possible conditions. This conversational approach transforms
free-text patient inputs into structured clinical insights, forming the first step of the AI-driven
patient journey.
</p>
</div>

</div>
""", unsafe_allow_html=True)

# ---------------------------
# API keys (Streamlit secrets or .env)
# ---------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# ---------------------------
# Gemini + Groq helpers
# ---------------------------
import google.generativeai as genai
from groq import Groq, PermissionDeniedError, APIConnectionError

gemini_model_obj = None
groq_client = None


def ensure_gemini():
    global gemini_model_obj
    if gemini_model_obj is not None:
        return gemini_model_obj
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model_obj = genai.GenerativeModel("gemini-2.5-flash")
    return gemini_model_obj


def ensure_groq():
    global groq_client
    if groq_client is not None:
        return groq_client
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    groq_client = Groq(api_key=GROQ_API_KEY)
    return groq_client


# ---------------------------
# Model Wrappers
# ---------------------------
def chat_with_gemini_messages(messages: list) -> str:
    model = ensure_gemini()
    prompt_parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        else:
            prompt_parts.append(f"Assistant: {content}")
    prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
    out = model.generate_content(prompt)
    reply = getattr(out, "text", "") or "I couldn't generate a safe response."
    return str(reply).strip()


def chat_with_groq_messages(messages: list) -> str:
    client = ensure_groq()
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=0.25,
        )
        return str(resp.choices[0].message.content).strip()
    except PermissionDeniedError:
        return "Groq permission issue."
    except APIConnectionError:
        return "Groq network error."


def generate_reply(model_choice: str, messages: list) -> str:
    if model_choice.startswith("Gemini"):
        return chat_with_gemini_messages(messages)
    return chat_with_groq_messages(messages)


# ---------------------------
# Utility
# ---------------------------
def ask_model(model_choice: str, system_prompt: str, user_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return generate_reply(model_choice, messages)


# ---------------------------
# Streamlit UI
# ---------------------------
#st.set_page_config(page_title="Start Your Consultation", page_icon="💊")
st.markdown('<div class="landing-wrapper">', unsafe_allow_html=True)
if st.session_state.show_intro:
    st.title("💊 Start Your Consulatation Here")
    st.write(
        "Ask general questions about symptoms or home care. "
        "**This is not a replacement for a doctor.**"
    )

# ---------------- PATIENT SELECTION (STEP 2) ----------------
import pandas as pd

PATIENT_FILE = "patients.csv"

if os.path.exists(PATIENT_FILE):
    patients_df = pd.read_csv(PATIENT_FILE)

    st.markdown("### 👤 Select Patient for This Consultation")

    selected_patient = st.selectbox(
        "Choose Patient",
        patients_df["patient_name"]
    )

    # Get selected patient row
    selected_row = patients_df[
        patients_df["patient_name"] == selected_patient
    ].iloc[0]

    # Store in session_state
    st.session_state.selected_patient_id = int(selected_row["patient_id"])
    st.session_state.selected_patient_name = selected_row["patient_name"]
    st.session_state.selected_patient_age = selected_row["age"]
    st.session_state.selected_patient_sex = selected_row["sex"]

else:
    st.error("❌ patients.csv not found.")

# model_choice = st.sidebar.selectbox("Choose model", ("Gemini", "Groq (Llama)"))

# TRIAGE STATES
if "show_triage" not in st.session_state:
    st.session_state.show_triage = False

if "triage_questions" not in st.session_state:
    st.session_state.triage_questions = []

if "triage_answers" not in st.session_state:
    st.session_state.triage_answers = []


# ---------------- Render Chat ----------------
for m in st.session_state["messages"]:
    if m["role"] == "system":
        continue

    if m["role"] == "user":
        st.markdown(
            f"""
            <div class="user-message-box">
                {m["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="ai-response-box">
                {m["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- TRIAGE READINESS INDICATOR ----------------
TRIAGE_WORD_THRESHOLD = 50

current_words = st.session_state.get("user_word_count", 0)
progress = min(current_words / TRIAGE_WORD_THRESHOLD, 1.0)

#st.markdown("### 🧭 Triage Readiness")

#st.progress(progress)

#if current_words < TRIAGE_WORD_THRESHOLD:
    #st.info(
       # f"📝 Gathering symptom details: **{current_words} / {TRIAGE_WORD_THRESHOLD} words collected**\n\n"
        #"Continue describing your symptoms so the assistant can understand your condition fully "
        #"before generating a clinical triage."
    #)
#else:
   # st.success(
       # "✅ Enough information collected. You can now generate a detailed clinical triage report."
    # )

# ---------------- Chat Input ----------------
col1, col2 = st.columns([3,1])

with col1:
    if st.session_state.show_intro:
        st.markdown("### Start Your Consultation")

with col2:
    model_choice = st.selectbox(
        "Choose Model",
        ("Gemini", "Groq (Llama)"),
        index=0
    )

user_input = st.chat_input("Describe your symptoms...")

if user_input:
    # 🔹 STEP 2: count user words
    word_count = len(user_input.split())
    st.session_state.user_word_count += word_count

    # store user message (NO change in logic)
    st.session_state["messages"].append(
        {"role": "user", "content": user_input}
    )
    st.session_state.show_intro = False
    st.session_state.show_triage = False
    st.session_state.triage_questions = []
    st.session_state.triage_answers = []
    st.session_state.triage_id = str(len(st.session_state["messages"]))

    # assistant response
    with st.spinner("Thinking..."):
        reply = generate_reply(model_choice, st.session_state["messages"])

    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.session_state.last_assistant_reply = reply


    # trigger triage
    #st.session_state.show_triage = True
    st.rerun()


# ---------------- TRIAGE INTRO + BUTTON ----------------
TRIAGE_WORD_THRESHOLD = 50

if (
    "last_assistant_reply" in st.session_state
    and st.session_state.last_assistant_reply
    and st.session_state.user_word_count >= TRIAGE_WORD_THRESHOLD
):

   
    # 🔘 Button (ONLY shown when ready)
    if st.button("🩺 Assess My Triage Summary"):

        # 🔹 Create session_id ONCE
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        triage_payload = {
            "messages": st.session_state["messages"],
            "last_assistant_reply": st.session_state["last_assistant_reply"],
            "model_choice": model_choice,
            "user_word_count": st.session_state["user_word_count"],
            "patient_id": int(st.session_state.selected_patient_id),
            "patient_name": str(st.session_state.selected_patient_name),
            "patient_age": int(st.session_state.selected_patient_age),
            "patient_sex": str(st.session_state.selected_patient_sex)
    }

        file_path = os.path.join(
            TRIAGE_DIR,
            f"{st.session_state.session_id}.json"
        )

        with open(file_path, "w") as f:
            json.dump(triage_payload, f, indent=2)

        st.switch_page("pages/Triage.py")

        #st.success("✅ Triage data prepared successfully")

        # 🔗 Build safe URL (local + cloud)
        #base_url = st.get_option("browser.serverAddress") or "localhost"
        #port = st.get_option("server.port")

        #triage_url = (
            #f"http://{base_url}:{port}/Triage"
            #f"?session_id={st.session_state.session_id}"
        

        #st.markdown(
          #f"""
          #<a href="/Triage?session_id={st.session_state.session_id}"
             #target="_blank"
             #style="
              # display:inline-block;
               #margin-top:16px;
               #font-size:18px;
               #font-weight:700;
               #color:#032B63;
          # ">
               #👉 Open Standalone Triage Report
            #</a>
            #""",
            #unsafe_allow_html=True
        #)


