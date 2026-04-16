from typing import Any, Dict, List
import os
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

from model_service import compare_vagmd_cases, predict_vagmd
from dotenv import load_dotenv
load_dotenv(override=True)
print("API KEY LOADED:", os.environ.get("OPENAI_API_KEY") is not None)

app = FastAPI(title="MDeveloper API")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def chat_with_openai(message: str) -> str:
    instructions = """
You are MDeveloper, a technical assistant for vacuum-enhanced air gap membrane distillation (VAGMD).

Your job is to help users interpret VAGMD performance, especially trends in:
- distillate flux
- condenser outlet temperature
- salinity
- feed inlet temperature
- condenser inlet temperature
- vacuum pressure
- channel velocity

Guidelines:
- Be technical, clear, and concise.
- Focus on VAGMD-specific reasoning.
- Do not invent exact numerical predictions unless model results are explicitly provided.
- If a question asks for quantitative comparison, say that model-backed comparison is needed.
- Keep answers to 150 words unless the user asks for more detail.
"""

    response = client.responses.create(
        model="gpt-5.4-nano",
        instructions=instructions,
        input=message,
        max_output_tokens=300,
    )

    # Debug prints
    print("=== OPENAI DEBUG START ===")
    print("message:", message)
    print("output_text repr:", repr(getattr(response, "output_text", None)))
    print("response id:", getattr(response, "id", None))
    print("response status:", getattr(response, "status", None))

    try:
        dump = response.model_dump()
        print("full response dump:")
        print(json.dumps(dump, indent=2, default=str))
    except Exception as exc:
        print("could not dump full response:", exc)
        print("raw response object:", response)

    # First try the convenience helper
    if getattr(response, "output_text", None):
        text = response.output_text.strip()
        if text:
            print("=== OPENAI DEBUG END (used output_text) ===")
            return text

    # Fallback: walk output items and collect text safely
    chunks = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)

    fallback_text = "\n".join(chunks).strip()
    print("fallback_text repr:", repr(fallback_text))
    print("=== OPENAI DEBUG END ===")

    if fallback_text:
        return fallback_text

    return "No text returned from OpenAI."


class VAGMDInput(BaseModel):
    T_mem_in: float
    T_con_in: float
    S: float
    v_chan: float
    vac: float


class VAGMDCase(BaseModel):
    name: str
    inputs: Dict[str, Any]


class CompareRequest(BaseModel):
    cases: List[VAGMDCase]


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "openai_key_loaded": os.environ.get("OPENAI_API_KEY") is not None,
    }


@app.post("/predict")
def predict(payload: VAGMDInput):
    try:
        return predict_vagmd(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/compare")
def compare(payload: CompareRequest):
    try:
        cases = [case.model_dump() for case in payload.cases]
        return compare_vagmd_cases(cases)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat")
def chat(payload: ChatRequest):
    try:
        reply = chat_with_openai(payload.message)
        return {"response": reply}
    except Exception as exc:
        print("chat error:", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc