import streamlit as st
from chatbot import ask_question, get_follow_up_suggestions
import os

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="Zenthic AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Custom CSS for better UI
# -----------------------
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    .user-message {
        text-align: right;
        background-color: #DCF8C6;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: auto;
    }
    .ai-message {
        text-align: left;
        background-color: #F1F0F0;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
        max-width: 70%;
    }
    .suggestion-button {
        background-color: #e3f2fd;
        border: 1px solid #90caf9;
        border-radius: 20px;
        padding: 8px 16px;
        margin: 4px;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Initialize Session State
# -----------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.title("⚙️ Zenthic AI Settings")
    st.markdown("---")
    
    # Model info
    st.subheader("Model Information")
    st.info("**Model:** GPT4All-J v1.3 Groovy")
    st.info(f"**CPU Threads:** {os.cpu_count()}")
    
    st.markdown("---")
    
    # Conversation controls
    st.subheader("Conversation Controls")
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.conversation = []
        st.session_state.suggestions = []
        st.rerun()
    
    if st.button("💾 Export Chat", use_container_width=True):
        if st.session_state.conversation:
            chat_export = "\n\n".join([
                f"{'User' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
                for msg in st.session_state.conversation
            ])
            st.download_button(
                label="Download as TXT",
                data=chat_export,
                file_name="zenthic_ai_conversation.txt",
                mime="text/plain"
            )
        else:
            st.warning("No conversation to export!")
    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 Stats")
    st.metric("Messages", len(st.session_state.conversation))
    user_msgs = len([m for m in st.session_state.conversation if m["role"] == "user"])
    ai_msgs = len([m for m in st.session_state.conversation if m["role"] == "ai"])
    st.metric("User Messages", user_msgs)
    st.metric("AI Responses", ai_msgs)

# -----------------------
# Main Chat Interface
# -----------------------
st.title("🤖 Zenthic AI Chatbot")
st.markdown("*Powered by GPT4All-J Groovy Model*")
st.markdown("---")

# Chat container
chat_container = st.container()

with chat_container:
    if not st.session_state.conversation:
        st.info("👋 Welcome! Ask me anything to get started.")
    else:
        # Display conversation history
        for msg in st.session_state.conversation:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='user-message'>🧑 {msg['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='ai-message'>🤖 {msg['content']}</div>",
                    unsafe_allow_html=True
                )

# Input section
st.markdown("---")
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Type your message:",
        key="user_input",
        placeholder="Ask me anything...",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send 📤", use_container_width=True)

# Handle user input
if send_button and user_input.strip():
    with st.spinner("🤔 Thinking..."):
        # Get AI response
        response = ask_question(user_input.strip())
        
        # Generate follow-up suggestions
        get_follow_up_suggestions(user_input.strip())
    
    # Clear input and refresh
    st.rerun()

# Display follow-up suggestions
if st.session_state.suggestions:
    st.markdown("### 💡 Suggested Follow-up Questions:")
    
    cols = st.columns(3)
    for i, suggestion in enumerate(st.session_state.suggestions[:3]):
        with cols[i % 3]:
            if st.button(
                suggestion,
                key=f"suggestion_{i}",
                use_container_width=True
            ):
                with st.spinner("🤔 Thinking..."):
                    # Process the selected suggestion
                    ask_question(suggestion)
                    get_follow_up_suggestions(suggestion)
                
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Built with ❤️ using Streamlit & GPT4All</div>",
    unsafe_allow_html=True
)