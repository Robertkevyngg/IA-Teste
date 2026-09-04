"""
API do IA Teste — IDS de rede.

    uvicorn api.main:app --reload
    abra http://127.0.0.1:8000

Endpoints:
    GET  /            dashboard
    GET  /health      status e modo (real ou demo)
    GET  /stream      fluxos classificados em tempo real (SSE)
    POST /predict     classifica uma lista de fluxos
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
WEB = ROOT / "web"

app = FastAPI(title="IA Teste — IDS", version="0.2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

state: dict[str, Any] = {"mode": "demo"}

# Classes do CIC-IDS2017 usadas no modo demo, com peso aproximado ao real:
# trafego benigno domina, ataques sao minoria.
DEMO_CLASSES = [
    ("BENIGN", 0.955, "none"),
    ("DoS Hulk", 0.014, "high"),
    ("PortScan", 0.012, "medium"),
    ("DDoS", 0.008, "high"),
    ("FTP-Patator", 0.004, "medium"),
    ("SSH-Patator", 0.003, "medium"),
    ("Web Attack - Brute Force", 0.002, "high"),
    ("Bot", 0.001, "high"),
    ("Infiltration", 0.001, "critical"),
]
SEVERITY = {name: sev for name, _, sev in DEMO_CLASSES}
# Severidade das demais classes reais do CIC-IDS2017 (o modo demo usa so um subconjunto).
SEVERITY.update({
    "BENIGN": "none",
    "DoS GoldenEye": "high", "DoS Slowhttptest": "high", "DoS slowloris": "high",
    "Heartbleed": "critical", "Infiltration": "critical",
    "Web Attack - XSS": "high", "Web Attack - Sql Injection": "critical",
})


@app.on_event("startup")
def load_models() -> None:
    path = MODELS / "supervised.joblib"
    if path.exists():
        import joblib
        state["supervised"] = joblib.load(path)
        state["mode"] = "real"
        print(f"modelo carregado ({len(state['supervised']['features'])} features) — modo real")
    else:
        print("models/supervised.joblib ausente — rodando em MODO DEMO com fluxos simulados")

    replay = ROOT / "data" / "processed" / "test.parquet"
    if replay.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(replay)
            # amostra embaralhada: o parquet vem ordenado por dia e classe, entao
            # ler em ordem mostraria so trafego benigno de segunda por varios minutos.
            n = min(len(df), 50_000)
            state["replay"] = df.sample(n=n, random_state=42).reset_index(drop=True)
            if state["mode"] == "demo":
                state["mode"] = "replay"
            print(f"replay carregado: amostra de {n:,} de {len(df):,} fluxos de teste")
        except Exception as exc:  # pragma: no cover
            print(f"nao consegui carregar o replay: {exc}")


# ----------------------------------------------------------------- geradores

def demo_flow() -> dict:
    """Um fluxo sintetico plausivel, para o dashboard rodar sem modelo."""
    names = [c[0] for c in DEMO_CLASSES]
    weights = [c[1] for c in DEMO_CLASSES]
    label = random.choices(names, weights=weights, k=1)[0]
    attack = label != "BENIGN"

    return {
        "ts": time.time(),
        "src": f"192.168.10.{random.randint(2, 60)}",
        "dst": f"172.16.0.{random.randint(1, 20)}",
        "port": random.choice([80, 443, 22, 21, 3389, 8080]) if not attack
                else random.choice([22, 21, 80, 445, random.randint(1, 65535)]),
        "duration_ms": round(random.expovariate(1 / 400), 1),
        "packets": random.randint(2, 40) if not attack else random.randint(1, 900),
        "bytes": random.randint(120, 9000) if not attack else random.randint(40, 220000),
        "label": label,
        "confidence": round(random.uniform(0.71, 0.999), 3),
        "severity": SEVERITY[label],
        "is_attack": attack,
        "source": "demo",
    }


def replay_flow(df, idx: int) -> dict:
    """Um fluxo real do conjunto de teste, classificado pelo modelo se houver."""
    row = df.iloc[idx % len(df)]
    label = str(row.get("label", "BENIGN"))
    confidence = 1.0
    source = "replay"

    if "supervised" in state:
        bundle = state["supervised"]
        X = np.array([[float(row.get(f, 0.0)) for f in bundle["features"]]])
        proba = bundle["model"].predict_proba(X)[0]
        label = str(bundle["label_encoder"].inverse_transform([proba.argmax()])[0])
        confidence = round(float(proba.max()), 3)
        source = "modelo"

    return {
        "ts": time.time(),
        "src": "-", "dst": "-",
        "port": int(row.get("Destination Port", row.get("Dst Port", 0)) or 0),
        "duration_ms": round(float(row.get("Flow Duration", 0)) / 1000, 1),
        "packets": int(row.get("Total Fwd Packets", 0) or 0),
        "bytes": int(row.get("Total Length of Fwd Packets", 0) or 0),
        "label": label,
        "confidence": confidence,
        "severity": SEVERITY.get(label, "medium" if label != "BENIGN" else "none"),
        "is_attack": label != "BENIGN",
        "source": source,
    }


def flow_iterator() -> Iterator[dict]:
    i = 0
    while True:
        if "replay" in state:
            yield replay_flow(state["replay"], i)
        else:
            yield demo_flow()
        i += 1


# ----------------------------------------------------------------- endpoints

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "mode": state["mode"],
        "model_loaded": "supervised" in state,
        "replay_loaded": "replay" in state,
    }


@app.get("/stream")
async def stream(rate: int = 12):
    """Server-Sent Events: `rate` fluxos por segundo."""
    rate = max(1, min(rate, 200))
    gen = flow_iterator()

    async def events():
        while True:
            batch = [next(gen) for _ in range(max(1, rate // 4))]
            yield f"data: {json.dumps(batch)}\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


class Flow(BaseModel):
    features: dict[str, float] = Field(..., description="features do CICFlowMeter")


class PredictRequest(BaseModel):
    flows: list[Flow]


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if "supervised" not in state:
        raise HTTPException(503, "modelo nao carregado — rode src/train_supervised.py")
    if not req.flows:
        raise HTTPException(400, "nenhum fluxo enviado")

    bundle = state["supervised"]
    feats, model, le = bundle["features"], bundle["model"], bundle["label_encoder"]

    started = time.perf_counter()
    X = np.array([[f.features.get(name, 0.0) for name in feats] for f in req.flows])
    proba = model.predict_proba(X)
    labels = le.inverse_transform(proba.argmax(axis=1))
    latency = (time.perf_counter() - started) * 1000

    return {
        "predictions": [
            {"label": str(l), "confidence": float(p.max()), "is_attack": str(l) != "BENIGN"}
            for l, p in zip(labels, proba)
        ],
        "latency_ms": round(latency, 2),
    }


@app.get("/")
def dashboard():
    index = WEB / "index.html"
    if not index.exists():
        raise HTTPException(404, "web/index.html nao encontrado")
    return FileResponse(index)
