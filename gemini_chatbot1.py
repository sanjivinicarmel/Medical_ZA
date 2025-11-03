import streamlit as st
import google.generativeai as genai
import time
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Medical Chatbot - Gemini (with Image Analysis)",
    page_icon="🏥",
    layout="centered"
)

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# System instruction
SYSTEM_INSTRUCTION = """You are a helpful and empathetic medical assistant chatbot.
You can analyze both **text** and **medical images** (like rashes, wounds, eye redness, swelling).
When an image is uploaded:
- Describe visible findings (color, shape, pattern, swelling, discharge, redness, etc.)
- Ask follow-up questions about associated symptoms (itching, pain, fever, etc.)
- Do NOT give a diagnosis.
- Always remind users that you are an AI assistant, not a real doctor, and recommend professional evaluation for any concern.
"""

MODEL_NAME = "gemini-2.0-flash-exp"

# Initialize session
if "chat" not in st.session_state:
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION
    )
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.messages = []

# App header
st.title("🏥 Medical Assistant Chatbot")
st.markdown("*Now supports image uploads!*")
st.caption("⚠️ This is NOT a substitute for professional medical advice.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- IMAGE UPLOAD ---
uploaded_image = st.file_uploader("📸 Upload a medical image (optional)", type=["jpg", "jpeg", "png"])

image_data = None
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_data = image  # Store for Gemini input

# --- CHAT INPUT ---
if prompt := st.chat_input("Describe your symptoms or image details..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # --- GEMINI RESPONSE ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Build input
            if image_data:
                # Pass both text and image
                response = st.session_state.chat.send_message(
                    [prompt, image_data],
                    stream=True
                )
            else:
                # Text only
                response = st.session_state.chat.send_message(prompt, stream=True)

            # Stream the response
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)

            message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"❌ **Error:** {str(e)}\n\nCheck your API key or file format."
            message_placeholder.markdown(full_response)

    # Add assistant reply to chat
    if full_response and not full_response.startswith("❌"):
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- SIDEBAR ---
with st.sidebar:
    st.header("🤖 Model Info")
    st.info(f"""
    **Currently using:**
    `{MODEL_NAME}`

    Gemini 2.0 Flash can understand:
    - 🧠 Text
    - 🖼️ Images (JPG, PNG)
    - ⚡ Extremely fast and accurate
    """)

    st.divider()
    st.header("💬 Example Inputs")
    st.markdown("""
    - "I have a rash on my arm"
    - "I feel pain near this wound" (upload image)
    - "My eye looks red" (upload image)
    """)

    if st.button("🔄 Clear Chat History", use_container_width=True):
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_INSTRUCTION
        )
        st.session_state.chat = model.start_chat(history=[])
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Powered by Google Gemini 2.0 Flash 🧠")
    st.caption("Built with ❤️ using Streamlit")
