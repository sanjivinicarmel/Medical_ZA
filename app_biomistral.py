import streamlit as st
import torch
import gc # <--- Added the 'gc' import for memory management
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

st.set_page_config(page_title="Medical Chatbot - BioMistral", page_icon="🏥")

st.title("🏥 Medical Chatbot - BioMistral 7B (4-bit)")

# --- 1. UPDATE MODEL PATH ---
model_path = "./models/biomistral" # This should match your download location

# Load model using session_state
if 'model' not in st.session_state:
    # --- UPDATE SPINNER MESSAGE ---
    with st.spinner("🔄 Loading BioMistral 7B model (4-bit, NF4)... Please wait"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                local_files_only=True,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # --- 2. SWITCH TO 4-BIT QUANTIZATION CONFIG ---
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,                   # <--- Key change: Use 4-bit
                bnb_4bit_quant_type="nf4",           # <--- Use the NF4 quantization scheme
                bnb_4bit_compute_dtype=torch.bfloat16, # <--- Use bfloat16 for faster 4-bit computation
                bnb_4bit_use_double_quant=True
            )
            
            # --- 3. LOAD MODEL WITH NEW 4-BIT CONFIG ---
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16 # <--- Recommended to ensure compatibility
            )
            
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            
            st.success("✅ BioMistral 7B (4-bit) loaded successfully!")
            # st.rerun() # Keep commented unless necessary for production environment
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.exception(e)
            st.stop()
            
# --- REST OF THE CODE REMAINS THE SAME ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Info")
    st.info("""
    **Model:** BioMistral 7B (4-bit QLoRA) 
    **Source:** Local Files 
    **GPU:** NVIDIA RTX 4050 (6GB)
    """)
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        st.metric("GPU Memory Used", f"{gpu_memory:.2f} GB")
    
    st.subheader("💬 Settings")
    max_tokens = st.slider("Max Response Length", 50, 300, 150)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
    
    # --- NEW: UNLOAD MODEL BUTTON TO FREE VRAM ---
    if st.button("🔌 Unload Model (Free VRAM)", use_container_width=True):
        if 'model' in st.session_state:
            # Delete the objects to allow memory to be released
            del st.session_state.model
            del st.session_state.tokenizer
            
            # Force a garbage collection to ensure objects are marked for deletion
            gc.collect() 
            
            # Clear the VRAM cache
            torch.cuda.empty_cache()
            st.info("Model unloaded. VRAM is now free. The model will reload when you next chat.")
            st.rerun() 
        else:
            st.warning("Model is already unloaded.")
    # ---------------------------------------------
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        torch.cuda.empty_cache()
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your medical question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            # Build conversation context (last 2 exchanges only)
            recent_messages = st.session_state.messages[-5:] if len(st.session_state.messages) > 5 else st.session_state.messages
            
            # Create a concise conversation history
            context = ""
            for msg in recent_messages[:-1]:  # Exclude the current user message
                if msg["role"] == "user":
                    context += f"Patient: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    context += f"Doctor: {msg['content']}\n"
            
            # Format prompt - Mistral chat template
            # BioMistral is based on Mistral-Instruct, so this template is correct
            formatted_prompt = f"<s>[INST] You are a helpful medical assistant. Answer the patient's question directly and clearly.\n{context}\n{prompt} [/INST]"
            
            # Tokenize
            inputs = st.session_state.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            # Move inputs to the correct device (GPU)
            inputs = {k: v.to(st.session_state.model.device) for k, v in inputs.items()}
            
            # Generate with stopping criteria
            outputs = st.session_state.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=40,
                do_sample=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=st.session_state.tokenizer.eos_token_id,
                eos_token_id=st.session_state.tokenizer.eos_token_id
            )
            
            # Decode response
            full_response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only new response
            if "[/INST]" in full_response:
                response = full_response.split("[/INST]")[-1].strip()
            else:
                response = full_response[len(formatted_prompt):].strip()
            
            # Remove any end tokens
            response = response.replace("</s>", "").strip()
            
            # Clean up response - stop at first complete sentence or paragraph
            if "\n\n" in response:
                response = response.split("\n\n")[0].strip()
            
            # Remove meta-commentary (common in small models)
            stop_phrases = [
                "to provide accurate",
                "as a medical assistant",
                "your medical assistant should",
                "here are some guidelines",
                "follow these guidelines",
                "seek medical attention"
            ]
            
            for phrase in stop_phrases:
                if phrase in response.lower():
                    response = response[:response.lower().index(phrase)].strip()
                    break
            
            # If response is empty or too generic, provide fallback
            if not response or len(response) < 10:
                response = "I understand you're experiencing discomfort. Could you describe your symptoms in more detail? For example, where is the pain located, how severe is it, and when did it start?"
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.sidebar.markdown("---")
st.sidebar.warning("⚠️ **Note:** For medical emergencies, always consult a real doctor.")
st.sidebar.caption("💡 BioMistral 7B is fine-tuned for biomedical tasks")