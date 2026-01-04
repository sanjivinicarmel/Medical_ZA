import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv(".env")

st.set_page_config(page_icon="💊", page_title="Medical Assistant", layout="wide")

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
</style>
""", unsafe_allow_html=True)


# ---------------- LANDING CONTENT ----------------
st.markdown("""
<div class="landing-wrapper">

<div class="top-box">
<h1 class="big-title">🩺 AI Symptom Intake — Intelligent Clinical Triage Assistant</h1>
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
st.title("💊 Start Your Consulatation Here")
st.write("Ask general questions about symptoms or home care. **This is not a replacement for a doctor.**")

# model_choice = st.sidebar.selectbox("Choose model", ("Gemini", "Groq (Llama)"))

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]

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
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.write(m["content"])

st.markdown("---")

# ---------------- Chat Input ----------------
col1, col2 = st.columns([3,1])

with col1:
    st.markdown("### Start Your Consultation")

with col2:
    model_choice = st.selectbox(
        "Choose Model",
        ("Gemini", "Groq (Llama)"),
        index=0
    )

user_input = st.chat_input("Describe your symptoms...")

if user_input:
    # user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
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
    st.session_state.show_triage = True
    st.rerun()



# ---------------- TRIAGE REPORT BUTTON ----------------
if "last_assistant_reply" in st.session_state and st.session_state.last_assistant_reply:

    if st.button("🩺 Generate Triage Report"):
        st.session_state.generate_triage = True


# ---------------- TRIAGE GENERATION ----------------
if st.session_state.get("generate_triage", False):

    # Extract latest user symptom
    user_symptom = ""
    for m in reversed(st.session_state["messages"]):
        if m["role"] == "user":
            user_symptom = m["content"]
            break

    triage_prompt = f"""
You are a professional clinical triage assistant.

Based on the conversation below, generate a structured TRIAGE REPORT.

Patient Statement:
{user_symptom}

Assistant Explanation:
{st.session_state.last_assistant_reply}

Your Output MUST be in this exact structure:

### 🩺 AI Triage Summary
• 2–3 sentence summary of situation

### ⚠️ Risk Level
One of:
LOW | MODERATE | HIGH | EMERGENCY

### 🔍 Key Symptoms Detected
- Bullet list

### 🏠 Suggested Home Remedies (if safe)
- Bullet list

### 👨‍⚕️ When to See a Doctor
Short guidance

### 🚑 Emergency Indicators
Clear bullet list of red flags

Be responsible, do NOT diagnose diseases.
"""

    triage_result = generate_reply(
        model_choice,
        [
            {"role": "system", "content": "You are an expert hospital clinical triage system. Be clear, calm, medically safe, responsible, and structured."},
            {"role": "user", "content": triage_prompt},
        ],
    )

    st.markdown("### 🩻 Triage Report")
    st.info(triage_result)
    st.warning("⚠️ This tool is for informational purposes only and not a medical diagnosis.")



# 🔽 CLOSE layout wrapper
st.markdown('</div>', unsafe_allow_html=True)
