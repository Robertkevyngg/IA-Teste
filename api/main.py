"""
API do IA Teste.

    uvicorn api.main:app --reload

Endpoints:
    GET  /health    status e modelos carregados
    POST /predict   classifica uma lista de fluxos
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODELS = Path("models")
app = FastAPI(title="IA Teste — IDS", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

state: dict[str, Any] = {}


@app.on_event("startup")
def load_models() -> None:
    path = MODELS / "supervised.joblib"
    if path.exists():
        state["supervised"] = joblib.load(path)
        print(f"modelo supervisionado carregado ({len(state['supervised']['features'])} features)")
    else:
        print("aviso: models/supervised.joblib nao encontrado — rode o treino primeiro")


class Flow(BaseModel):
    features: dict[str, float] = Field(..., description="features do CICFlowMeter")


class PredictRequest(BaseModel):
    flows: list[Flow]


class Prediction(BaseModel):
    label: str
    confidence: float
    is_attack: bool


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "supervised": "supervised" in state}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if "supervised" not in state:
        raise HTTPException(503, "modelo nao carregado")
    if not req.flows:
        raise HTTPException(400, "nenhum fluxo enviado")

    bundle = state["supervised"]
    feats, model, le = bundle["features"], bundle["model"], bundle["label_encoder"]

    started = time.perf_counter()
    X = np.array([[f.features.get(name, 0.0) for name in feats] for f in req.flows])
    proba = model.predict_proba(X)
    labels = le.inverse_transform(proba.argmax(axis=1))
    latency = (time.perf_counter() - started) * 1000

    return PredictResponse(
        predictions=[
            Prediction(label=str(label), confidence=float(p.max()),
                       is_attack=str(label) != "BENIGN")
            for label, p in zip(labels, proba)
        ],
        latency_ms=round(latency, 2),
    )
