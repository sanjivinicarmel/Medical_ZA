# app.py (cleaned + defensive dedupe of trailing assistant messages)
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv(".env")

# ---------------------------
# API keys (Streamlit secrets or .env)
# ---------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# ---------------------------
# Gemini + Groq helpers (kept from your code)
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
# System prompt: conversational + ask a short follow-up question at the end
# ---------------------------
SYSTEM_PROMPT = (
    "You are a friendly, helpful medical chat assistant. Speak naturally like ChatGPT.\n"
    "Guidelines:\n"
    "- Use conversational language (short paragraphs or bullets if needed).\n"
    "- Ask 1 short, gentle clarifying question at the end of your reply to invite more details,\n"
    "  e.g. 'Can you tell me more about your symptoms?' or 'When did this start?'.\n"
    "- Offer simple non-prescriptive suggestions when appropriate.\n"
    "- Recommend seeing a doctor only if symptoms are severe, sudden, spreading, or persistent.\n"
    "- Never give a formal medical diagnosis or prescribe medication.\n"
    "- If health advice is discussed, end with: 'This is general information and not a substitute for professional medical advice.'\n"
)

# ---------------------------
# Model call helpers using message history
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
        return (
            "I couldn't reach the Groq model due to a permission issue. "
            "Try switching the model or network. "
            "This is general information and not a substitute for professional medical advice."
        )
    except APIConnectionError:
        return (
            "I couldn't connect to Groq due to a connectivity error. "
            "Please try again later. "
            "This is general information and not a substitute for professional medical advice."
        )


def generate_reply(model_choice: str, messages: list) -> str:
    if model_choice.startswith("Gemini"):
        return chat_with_gemini_messages(messages)
    return chat_with_groq_messages(messages)


# ---------------------------
# Defensive dedupe utilities
# ---------------------------
def dedupe_consecutive_assistant_messages(messages: list, keep_last_n=1) -> list:
    """
    Removes extra identical consecutive assistant messages at the end of the list,
    keeping at most `keep_last_n` copies of the identical trailing assistant message.
    Also collapses any immediate consecutive duplicates throughout the list.
    """
    if not messages:
        return messages

    # First, collapse any immediate consecutive duplicates throughout the list
    collapsed = []
    prev = None
    for m in messages:
        if prev and m["role"] == "assistant" and prev["role"] == "assistant" and m["content"] == prev["content"]:
            # skip duplicate assistant message immediately following an identical assistant
            continue
        collapsed.append(m)
        prev = m

    # Then ensure the trailing assistant messages are at most keep_last_n copies
    # (rare case, but safe)
    # Count trailing assistant messages with identical content
    i = len(collapsed) - 1
    if i >= 0 and collapsed[i]["role"] == "assistant":
        last_content = collapsed[i]["content"]
        count = 1
        j = i - 1
        while j >= 0 and collapsed[j]["role"] == "assistant" and collapsed[j]["content"] == last_content:
            count += 1
            j -= 1
        if count > keep_last_n:
            # remove extras
            to_remove = count - keep_last_n
            for _ in range(to_remove):
                collapsed.pop()

    return collapsed


# ---------------------------
# Streamlit UI: chat-like flow
# ---------------------------
st.set_page_config(page_title="Medical Chat Assistant", page_icon="💊")
st.title("💊 Medical Chat Assistant")
st.write(
    "Ask general questions about symptoms or home care. "
    "**This is not a replacement for a doctor.**"
)

model_choice = st.sidebar.selectbox("Choose model", ("Gemini", "Groq (Llama)"))
st.sidebar.markdown("### About")
st.sidebar.write(
    "- Conversational assistant\n"
    "- Asks clarifying questions when needed\n"
    "- Provides general information only"
)

# Initialize messages if missing
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]

# Defensive dedupe on load (prevents showing past duplicate replies)
st.session_state["messages"] = dedupe_consecutive_assistant_messages(st.session_state["messages"], keep_last_n=1)

# Reset conversation
if st.sidebar.button("Reset conversation"):
    st.session_state["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]

# Render chat using st.chat_message if available, otherwise markdown fallback
use_chat_api = hasattr(st, "chat_message")


def render_messages(messages):
    # skip system message when rendering
    for m in messages:
        if m["role"] == "system":
            continue
        if use_chat_api:
            with st.chat_message("user" if m["role"] == "user" else "assistant"):
                st.write(m["content"])
        else:
            if m["role"] == "user":
                st.markdown(f"**You:** {m['content']}")
            else:
                st.markdown(f"**Assistant:**  \n{m['content']}")


# Render existing conversation
render_messages(st.session_state["messages"])

st.markdown("---")


# Utility: detect if assistant reply already contains a follow-up question
def ends_with_question(text: str) -> bool:
    text = text.strip()
    if text.endswith("?"):
        return True
    lowered = text.lower()
    for phrase in ("tell me more", "can you tell", "could you tell", "please describe", "please tell"):
        if phrase in lowered:
            return True
    return False


# Chat input form: empty on render and cleared on submit
with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_area(
        "Type your message here:",
        value="",
        height=120,
        placeholder="e.g. I applied Dettol on my skin — will it help?",
        key="user_input",
    )
    submitted = st.form_submit_button("Send")

if submitted:
    if not user_text or not user_text.strip():
        st.warning("Please type a message.")
    else:
        # Append the user's message to history (once)
        st.session_state["messages"].append({"role": "user", "content": user_text.strip()})

        # Generate assistant reply using full history
        with st.spinner("Thinking..."):
            try:
                assistant_reply = generate_reply(model_choice, st.session_state["messages"])
            except Exception:
                assistant_reply = (
                    "Sorry — I couldn't reach the model right now. "
                    "Please try again in a moment."
                )

        # Ensure assistant reply includes a friendly follow-up question
        if not ends_with_question(assistant_reply):
            if "this is general information" in assistant_reply.lower():
                parts = assistant_reply.rsplit("\n", 1)
                if len(parts) == 2 and "this is general information" in parts[-1].lower():
                    assistant_reply = (
                        parts[0].rstrip()
                        + "\n\nCould you tell me a bit more about your symptoms "
                        "(when they started, severity, and any other symptoms)?\n\n"
                        + parts[1]
                    )
                else:
                    assistant_reply = assistant_reply.rstrip() + "\n\nCould you tell me a bit more about your symptoms (when they started, severity, and any other symptoms)?"
            else:
                assistant_reply = assistant_reply.rstrip() + "\n\nCould you tell me a bit more about your symptoms (when they started, severity, and any other symptoms)?"

        # Prevent duplicate appends: add assistant reply only if it's not already the last assistant message
        last_msg = st.session_state["messages"][-1] if st.session_state["messages"] else None
        if not (last_msg and last_msg.get("role") == "assistant" and last_msg.get("content") == assistant_reply):
            st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})

# UX hint
st.caption("Tip: after the assistant asks a follow-up question, use the same box to reply (this keeps the conversation flowing).")
