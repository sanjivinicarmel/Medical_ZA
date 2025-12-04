import os
from dotenv import load_dotenv

print(">>> Starting test_groq.py")

# Load .env from the same folder
load_dotenv(".env")

key = os.getenv("GROQ_API_KEY")
print("GROQ_API_KEY from env:", repr(key))

if not key:
    print("!! GROQ_API_KEY is missing. Check your .env file location and name.")
    raise SystemExit(1)

from groq import Groq

print(">>> Creating Groq client...")
client = Groq(api_key=key)

print(">>> Calling Groq chat completion...")
try:
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Hello from test_groq.py!"}],
    )
    print(">>> Groq replied:")
    print(resp.choices[0].message.content)
except Exception as e:
    import traceback
    print("!! Exception while calling Groq:")
    traceback.print_exc()
