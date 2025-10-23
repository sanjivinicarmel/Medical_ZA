import streamlit as st
from pygpt4all import GPT4All_J
import os

# -----------------------
# Backend setup
# -----------------------
os.environ["N_THREADS"] = str(os.cpu_count())

@st.cache_resource
def load_model():
    return GPT4All_J("models/ggml-gpt4all-j-v1.3-groovy.bin")

model = load_model()

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

# Export conversation_history (alias for conversation)
conversation_history = st.session_state.conversation

# -----------------------
# Functions
# -----------------------
def ask_question(user_input):
    """
    Generate AI response and store in session state.
    """
    st.session_state.conversation.append({"role": "user", "content": user_input})
    
    # Build the prompt from conversation history
    prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" 
                        for msg in st.session_state.conversation]) + "\nAI:"

    # Generate response (pygpt4all doesn't support streaming parameter)
    response = model.generate(prompt)
    
    # Handle if response is a generator or list
    if not isinstance(response, str):
        response = "".join(response)
    
    response = response.strip()
    st.session_state.conversation.append({"role": "ai", "content": response})
    return response

def get_follow_up_suggestions(user_input):
    """
    Generate 2-3 follow-up questions and store in session state.
    """
    prompt = f"Based on this conversation, suggest 3 short follow-up questions (one per line):\n"
    prompt += "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" 
                        for msg in st.session_state.conversation[-4:]])  # Last 2 exchanges
    prompt += "\n\nFollow-up questions:"
    
    suggestions = model.generate(prompt)
    
    # Handle if response is a generator or list
    if not isinstance(suggestions, str):
        suggestions = "".join(suggestions)

    # Parse suggestions
    suggestion_list = [s.strip() for s in suggestions.split("\n") if s.strip() and len(s.strip()) > 10]
    
    # Clean up suggestions (remove numbering, bullets, etc.)
    cleaned_suggestions = []
    for s in suggestion_list:
        # Remove common prefixes
        s = s.lstrip('123456789.-•*) ')
        if s and '?' in s:
            cleaned_suggestions.append(s)
    
    st.session_state.suggestions = cleaned_suggestions[:3]  # Limit to 3

# -----------------------
# Streamlit UI (only runs if this is the main file)
# -----------------------
if __name__ == "__main__":
    st.title("🧠 GPT4All J - Groovy Chatbot")

    # Display conversation as chat bubbles
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            st.markdown(f"<div style='text-align: right; background-color: #DCF8C6; padding:8px; border-radius:10px; margin:4px 0;'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; background-color: #F1F0F0; padding:8px; border-radius:10px; margin:4px 0;'>{msg['content']}</div>", unsafe_allow_html=True)

    # Input box
    user_input = st.text_input("Type your question here:", key="input_box")

    # Handle user input
    if user_input and st.button("Send"):
        response = ask_question(user_input)
        get_follow_up_suggestions(user_input)
        st.rerun()  # Refresh UI to show new conversation

    # Display follow-up suggestions
    if st.session_state.suggestions:
        st.markdown("**Follow-up questions:**")
        for i, suggestion in enumerate(st.session_state.suggestions[:3]):  # Limit to 3
            if st.button(suggestion, key=f"suggestion_{i}"):
                ask_question(suggestion)
                get_follow_up_suggestions(suggestion)
                st.session_state.suggestions = []
                st.rerun()