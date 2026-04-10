from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from model_service import predict_vagmd, compare_vagmd_cases

app = FastAPI()


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


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: VAGMDInput):
    return predict_vagmd(payload.model_dump())


@app.post("/compare")
def compare(payload: CompareRequest):
    cases = [case.model_dump() for case in payload.cases]
    return compare_vagmd_cases(cases)