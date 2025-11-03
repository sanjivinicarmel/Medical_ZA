import streamlit as st
from groq import Groq
import time

# Page config
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🏥",
    layout="centered"
)

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = get_groq_client()

# System prompt for medical assistant
SYSTEM_PROMPT = """You are a helpful and empathetic medical assistant chatbot. Your role is to:

1. **Ask Follow-up Questions**: When a user mentions symptoms (like chest pain), ask relevant follow-up questions about:
   - Severity (1-10 scale)
   - Duration (when did it start?)
   - Location (specific area)
   - Nature (sharp, dull, burning, pressure?)
   - Associated symptoms (shortness of breath, nausea, sweating, etc.)
   - Triggers (what makes it better or worse?)
   - Medical history (any pre-existing conditions?)

2. **Be Conversational**: Ask questions naturally, one or two at a time, not all at once.

3. **Show Empathy**: Be caring and professional in your responses.

4. **Important Disclaimers**: 
   - Always remind users that you're an AI assistant, not a real doctor
   - For serious symptoms (chest pain, difficulty breathing, severe pain), advise seeking immediate medical attention
   - Never provide definitive diagnoses

5. **Gather Information Systematically**: Build a complete picture before suggesting anything.

Remember: Your goal is to help users understand their symptoms better and guide them to appropriate care, not to replace professional medical advice."""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# App header
st.title("🏥 Medical Assistant Chatbot")
st.markdown("*An AI-powered assistant to help understand your symptoms*")
st.caption("⚠️ This is NOT a substitute for professional medical advice")

# Display chat history (excluding system message)
for message in st.session_state.messages[1:]:  # Skip system message
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Describe your symptoms... (e.g., 'My chest is hurting')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Call Groq API with the CORRECT model
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Updated model name
                messages=st.session_state.messages,
                temperature=0.7,
                max_tokens=1024,
                stream=True
            )
            
            # Stream the response
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            full_response = f"❌ Error: {str(e)}\n\nPlease check your API key in `.streamlit/secrets.toml`"
            message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar with info and controls
with st.sidebar:
    st.header("About")
    st.info("""
    This chatbot uses AI to help you understand your symptoms by asking relevant follow-up questions.
    
    **How to use:**
    1. Describe your symptoms
    2. Answer the follow-up questions
    3. Get guidance on next steps
    
    **Remember:** Always consult a real doctor for medical advice!
    """)
    
    st.header("Example Queries")
    st.markdown("""
    - "My chest is hurting"
    - "I have a headache"
    - "I feel dizzy and nauseous"
    - "I have a fever and cough"
    """)
    
    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        st.rerun()
    
    st.divider()
    st.caption("Powered by Groq + Llama 3.3")