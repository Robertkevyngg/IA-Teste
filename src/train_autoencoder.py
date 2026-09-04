"""
Detector de anomalia (camada B).

Treina um autoencoder SO em trafego benigno. Fluxo cujo erro de reconstrucao
passa do limiar calibrado e sinalizado como anomalo — inclusive ataques que
nao existiam no dataset de treino.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

BENIGN = "BENIGN"


class AutoEncoder(nn.Module):
    def __init__(self, n_features: int, latent: int = 12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def reconstruction_error(model, X: torch.Tensor, batch: int = 4096) -> np.ndarray:
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            chunk = X[i:i + batch]
            errs.append(((model(chunk) - chunk) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(errs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--percentile", type=float, default=99.0,
                   help="percentil do erro em benigno que define o limiar")
    args = p.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.processed / "train.parquet")
    test = pd.read_parquet(args.processed / "test.parquet")
    feats = [c for c in train.columns if c != "label"]

    benign = train[train["label"] == BENIGN][feats]
    scaler = StandardScaler().fit(benign)

    X = torch.tensor(scaler.transform(benign), dtype=torch.float32)
    split = int(len(X) * 0.9)
    X_tr, X_val = X[:split], X[split:]

    model = AutoEncoder(len(feats))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        total = 0.0
        for i in range(0, len(X_tr), 512):
            batch = X_tr[perm[i:i + 512]]
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        val = reconstruction_error(model, X_val).mean()
        print(f"epoch {epoch + 1:02d}  train {total / len(X_tr):.5f}  val {val:.5f}")

    # limiar calibrado no benigno de validacao
    threshold = float(np.percentile(reconstruction_error(model, X_val), args.percentile))

    X_te = torch.tensor(scaler.transform(test[feats]), dtype=torch.float32)
    err = reconstruction_error(model, X_te)
    flagged = err > threshold
    is_attack = (test["label"] != BENIGN).values

    recall = flagged[is_attack].mean()
    fpr = flagged[~is_attack].mean()
    print(f"\nlimiar (p{args.percentile}) : {threshold:.5f}")
    print(f"recall em ataques     : {recall:.4f}")
    print(f"FPR em benigno        : {fpr:.5f}")

    torch.save(model.state_dict(), args.models / "autoencoder.pt")
    joblib.dump({"scaler": scaler, "threshold": threshold, "features": feats},
                args.models / "autoencoder_meta.joblib")


if __name__ == "__main__":
    main()
