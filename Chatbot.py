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

if "user_word_count" not in st.session_state:
    st.session_state.user_word_count = 0

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

st.markdown("### 🧭 Triage Readiness")

st.progress(progress)

if current_words < TRIAGE_WORD_THRESHOLD:
    st.info(
        f"📝 Gathering symptom details: **{current_words} / {TRIAGE_WORD_THRESHOLD} words collected**\n\n"
        "Continue describing your symptoms so the assistant can understand your condition fully "
        "before generating a clinical triage."
    )
else:
    st.success(
        "✅ Enough information collected. You can now generate a detailed clinical triage report."
    )

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
    # 🔹 STEP 2: count user words
    word_count = len(user_input.split())
    st.session_state.user_word_count += word_count

    # store user message (NO change in logic)
    st.session_state["messages"].append(
        {"role": "user", "content": user_input}
    )
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




# ---------------- TRIAGE INTRO SECTION ----------------
# ---------------- TRIAGE INTRO + BUTTON ----------------
TRIAGE_WORD_THRESHOLD = 50

if (
    "last_assistant_reply" in st.session_state
    and st.session_state.last_assistant_reply
    and st.session_state.user_word_count >= TRIAGE_WORD_THRESHOLD
):

    st.markdown("""
    <div class="triage-title-box">
        🩺 Intelligent Clinical Triage Assistant
    </div>

    <div class="triage-desc-box">
        This triage is designed to help you understand the <b>next appropriate steps</b>
        based on your symptoms.
        <br><br>
        It does not diagnose conditions, but provides guidance on
        <b>severity</b>, <b>self-care options</b>, <b>when to consult a doctor</b>,
        and <b>warning signs</b> that may need urgent medical attention.
        <br><br>
        The goal is to support informed decision-making, reduce unnecessary anxiety,
        and help you seek the <b>right level of care at the right time</b>.
    </div>
    """, unsafe_allow_html=True)

    if st.button("🩺 Generate Triage Report"):
        st.session_state.generate_triage = True


# ---------------- TRIAGE GENERATION ----------------
if (
    st.session_state.get("generate_triage", False)
    and st.session_state.user_word_count >= TRIAGE_WORD_THRESHOLD
):

    # Extract latest user symptom
    user_symptom = ""
    for m in reversed(st.session_state["messages"]):
        if m["role"] == "user":
            user_symptom = m["content"]
            break

    triage_prompt = f"""
You are an experienced clinical triage assistant.

Your role is NOT to diagnose diseases.
Your role is to guide patients on what to do next in a safe, practical, and reassuring manner.

Think like a doctor explaining next steps to a patient in simple language.

-------------------------
PATIENT CONTEXT
-------------------------
Patient Message:
{user_symptom}

Assistant Explanation:
{st.session_state.last_assistant_reply}

-------------------------
TRIAGE INSTRUCTIONS
-------------------------
1. Do NOT diagnose or name diseases.
2. Use simple, non-technical, reassuring language.
3. Include ONLY sections that are relevant to this case.
4. If symptoms are mild:
   - Emphasize reassurance, self-care, and home-based recovery.
5. If symptoms are moderate:
   - Emphasize monitoring, supportive care, and when to consider medical advice.
6. If symptoms are severe:
   - Clearly explain urgency while maintaining a calm and supportive tone.
7. Provide practical, actionable recommendations the patient can realistically follow at home.
8. Home remedies and OTC advice must be conservative, optional, and non-prescriptive.
9. NEVER give medication dosages or prescription drugs.
10. Avoid alarming language; focus on reducing anxiety and empowering the patient.
11. Patient safety is the top priority at all times.

-------------------------
FORMATTING RULES (MANDATORY)
-------------------------
- Each section title MUST appear on its own line.
- Content MUST begin on the next line after the title.
- Use bullet points (-) for all content under each section.
- NEVER place sentences on the same line as a section title.
- Do NOT merge headings and sentences.
- Keep sections visually clean and easy to scan.


-------------------------
OUTPUT FORMAT
-------------------------

🩺 Triage Summary
- 2–3 sentences in plain language.

⚠️ Overall Assessment
- LOW | MODERATE | NEEDS MEDICAL REVIEW
- One calm sentence explaining why.

🔍 Key Symptoms Observed
- Bullet list only.

🏠 Home Care & Natural Remedies (if appropriate)
- Bullet list of practical actions.

💊 Optional Relief Measures (Optional)
- Bullet list.
- No dosages.

💙 Reassurance
- 1–2 calming, validating sentences.

🕒 What to Expect Over the Next 24–72 Hours
- Bullet list describing normal progression.

🔔 Important Changes to Watch For
- Bullet list.
- Use calm language (no panic).



-------------------------
FINAL SAFETY LINE (MANDATORY)
-------------------------
"This information is for general guidance only and not a substitute for professional medical care."
"""

    triage_result = generate_reply(
        model_choice,
        [
            {
                "role": "system",
                "content": "You are an expert hospital clinical triage system. Be clear, calm, medically safe, responsible, and structured."
            },
            {"role": "user", "content": triage_prompt},
        ],
    )

    st.markdown("### 🩻 Triage Report")
    st.info(triage_result)
    st.warning("⚠️ This tool is for informational purposes only and not a medical diagnosis.")
