import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured in .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Finger Millet GenAI API",
    description="Generative AI assistant for finger millet disease detection",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    disease: str

    confidence: float

    question: str


# ============================================================
# RESPONSE MODEL
# ============================================================

class ChatResponse(BaseModel):

    answer: str


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Finger Millet GenAI API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    prompt = f"""
You are an agricultural AI assistant specializing
in finger millet disease detection and management.

The user's detected condition is:

Disease:
{request.disease}

Model confidence:
{request.confidence * 100:.2f}%

The user asks:

{request.question}

Instructions:

1. Answer specifically about finger millet whenever possible.
2. Use simple and clear language.
3. Give practical agricultural guidance.
4. Explain technical terms when necessary.
5. Do not claim that the AI prediction is a laboratory-confirmed diagnosis.
6. If the confidence is low, remind the user that the prediction
   should be verified.
7. If the question is unrelated to finger millet disease or
   crop management, politely explain that you specialize in
   finger millet.
8. Do not invent pesticide doses or chemical application rates.
9. Recommend consulting a qualified agricultural expert when
   laboratory confirmation or locally approved treatment is required.

Provide a useful answer to the user's question.
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        answer = response.text

        if not answer:
            answer = (
                "The AI did not return an answer. "
                "Please try asking the question again."
            )

        return ChatResponse(
            answer=answer
        )

    except Exception as e:

        return ChatResponse(
            answer=(
                "The AI service is temporarily unavailable. "
                f"Please try again later. Error: {str(e)}"
            )
        )