import streamlit as st
import google.generativeai as genai
from groq import Groq

# --- PAGE CONFIG ---
st.set_page_config(page_title="Medical AI Assistant", page_icon="🧠", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose AI Model:", ["Gemini", "Groq"])
st.sidebar.markdown("---")
st.sidebar.info("Switch between **Gemini** (Google) and **Groq** (Llama).")

# --- LOAD SECRETS ---
st.write("Secrets available:", list(st.secrets.keys()))
try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Please add it in Streamlit → Settings → Secrets.")
    st.stop()

# --- MODEL INITIALIZATION ---
if model_choice == "Gemini":
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
else:
    client = Groq(api_key=GROQ_KEY)
    model = client.chat.completions

# --- MAIN UI ---
st.title("🩺 Medical Chat Assistant")
st.caption("Your AI-powered medical companion — using Gemini or Groq")

# --- CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_model" not in st.session_state:
    st.session_state.current_model = model_choice

# Reset chat history when model changes
if model_choice != st.session_state.current_model:
    st.session_state.messages = []
    st.session_state.current_model = model_choice
    st.info(f"🧠 Switched to **{model_choice}** model. Chat reset.")

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- USER INPUT ---
if prompt := st.chat_input("Ask me about symptoms, conditions, or medications..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- MODEL RESPONSE ---
    with st.chat_message("assistant"):
        with st.spinner(f"💭 Thinking with {model_choice}..."):
            if model_choice == "Gemini":
                response = model.generate_content(prompt)
                reply = response.text
            else:
                response = model.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                )
                reply = response.choices[0].message.content

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
