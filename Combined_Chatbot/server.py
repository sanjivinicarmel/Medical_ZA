# server.py
import os
import logging
import traceback
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from groq import PermissionDeniedError, APIConnectionError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(".env")

groq_client = None
gemini_model = None


# ---------------------------------------------
# Groq client
# ---------------------------------------------
def ensure_groq():
    global groq_client
    if groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(500, "GROQ_API_KEY is not set on the server")
        groq_client = Groq(api_key=api_key)
    return groq_client


# ---------------------------------------------
# Gemini client
# ---------------------------------------------
# ---------------------------------------------
# Gemini client
# ---------------------------------------------
def ensure_gemini():
    global gemini_model
    if gemini_model is not None:
        return gemini_model

    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(500, "GEMINI_API_KEY is not set on the server")

    genai.configure(api_key=api_key)

    # 👉 Pick ONE model that is clearly in your "available" list
    # From your logs we know these exist:
    #   'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-flash-latest', 'gemini-pro-latest', ...
    model_name = "gemini-2.5-flash"      # you can also try "gemini-flash-latest"

    print("Using Gemini model:", model_name)
    gemini_model = genai.GenerativeModel(model_name)
    return gemini_model




# ---------------------------------------------
# FastAPI app
# ---------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h2>index.html not found</h2>", status_code=404)
    return FileResponse(index_path)


@app.get("/health")
def health():
    return {"ok": True}


class ChatIn(BaseModel):
    message: str
    model: Literal["gemini", "groq"]
    system_prompt: Optional[str] = (
        "You are a concise health information explainer.\n"
        "Rules:\n"
        "- Answer ONLY in 4–7 bullet points that start with '-'.\n"
        "- No numbered lists.\n"
        "- No long intro or apology.\n"
        "- No paragraphs, each bullet 1–2 short sentences.\n"
        "- At the very end, add one last bullet: "
        "'This is general information and not a substitute for professional medical advice.'\n"
    )

@app.post("/api/chat")
def chat(payload: ChatIn):
    user = payload.message.strip()
    if not user:
        raise HTTPException(400, "Empty message")

    try:
        # ============================
        # GROQ MODEL
        # ============================
        if payload.model == "groq":
            from groq import PermissionDeniedError, APIConnectionError

            client = ensure_groq()
            try:
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",   # keep your model
                    messages=[
                        {"role": "system", "content": payload.system_prompt},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.3,
                )
                reply = str(resp.choices[0].message.content)

            except PermissionDeniedError:
                reply = (
                    "- I could not reach the Groq model due to a network or permission issue.\n"
                    "- This often happens when the internet connection or firewall blocks Groq.\n"
                    "- Try switching networks, using a hotspot, or choose the Gemini model.\n"
                    "- This is general information."
                )

            except APIConnectionError:
                reply = (
                    "- I could not connect to Groq due to an SSL or connectivity error.\n"
                    "- Try using a different internet connection or VPN.\n"
                    "- You can also switch to the Gemini model above.\n"
                    "- This is general information."
                )

        # ============================
        # GEMINI MODEL (your working version)
        # ============================
        else:
            model = ensure_gemini()
            gemini_prompt = (
                f"{payload.system_prompt}\n\n"
                f"User question:\n{user}\n\n"
                "Remember: follow the rules strictly and respond now in bullet points only."
            )

            out = model.generate_content(gemini_prompt)
            reply = getattr(out, "text", "") or "I couldn't generate a safe response."
            reply = str(reply)

        return {"reply": reply}

    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        logging.error(tb)
        raise HTTPException(500, f"Error generating response: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
