from typing import Any, Dict, List
import os
import json
import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

print("Starting api.py")
print("OPENAI_API_KEY loaded:", os.environ.get("OPENAI_API_KEY") is not None)

app = FastAPI(title="MDeveloper API")


def get_openai_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    return OpenAI(api_key=api_key)


def chat_with_openai(message: str) -> str:
    client = get_openai_client()

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
        model="gpt-4.1-mini",
        instructions=instructions,
        input=message,
        max_output_tokens=300,
    )

    print("=== OPENAI DEBUG START ===")
    print("message:", message)
    print("output_text repr:", repr(getattr(response, "output_text", None)))
    print("response id:", getattr(response, "id", None))
    print("response status:", getattr(response, "status", None))
    print("=== OPENAI DEBUG END ===")

    if getattr(response, "output_text", None):
        text = response.output_text.strip()
        if text:
            return text

    chunks = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)

    return "\n".join(chunks).strip() or "No text returned from OpenAI."


class VAGMDInput(BaseModel):
    T_mem_in: float
    T_con_in: float
    S: float
    v_chan: float
    vac: float
    L_type: int
    sp_type: int
    spa_type: int
    membrane: int


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
        print("=== PREDICT START ===")
        from model_service import predict_vagmd

        data = payload.model_dump()
        print("PREDICT INPUT:", json.dumps(data, indent=2))

        result = predict_vagmd(data)

        print("PREDICT RESULT:", result)
        print("=== PREDICT SUCCESS ===")
        return result

    except Exception as exc:
        print("=== PREDICT ERROR ===")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/compare")
def compare(payload: CompareRequest):
    try:
        print("=== COMPARE START ===")
        from model_service import compare_vagmd_cases

        cases = [case.model_dump() for case in payload.cases]
        print("COMPARE CASES:", json.dumps(cases, indent=2))

        result = compare_vagmd_cases(cases)

        print("COMPARE RESULT:", json.dumps(result, indent=2, default=str))
        print("=== COMPARE SUCCESS ===")
        return result

    except Exception as exc:
        print("=== COMPARE ERROR ===")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat")
def chat(payload: ChatRequest):
    try:
        print("=== CHAT START ===")
        reply = chat_with_openai(payload.message)
        print("=== CHAT SUCCESS ===")
        return {"response": reply}

    except Exception as exc:
        print("=== CHAT ERROR ===")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc