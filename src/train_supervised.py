"""
Classificador multiclasse de ataques (camada A).

Baseline de regressao logistica -> LightGBM. Reporta metricas por classe,
FPR e PR-AUC. Acurácia nao entra no relatorio de proposito.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder

BENIGN = "BENIGN"


def load(processed: Path):
    train = pd.read_parquet(processed / "train.parquet")
    test = pd.read_parquet(processed / "test.parquet")
    feats = [c for c in train.columns if c != "label"]
    return train, test, feats


def false_positive_rate(y_true_labels, y_pred_labels) -> float:
    """Fracao do trafego benigno classificada como ataque."""
    benign = y_true_labels == BENIGN
    if benign.sum() == 0:
        return float("nan")
    return float((y_pred_labels[benign] != BENIGN).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--models", type=Path, default=Path("models"))
    args = p.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)

    import lightgbm as lgb

    train, test, feats = load(args.processed)
    le = LabelEncoder().fit(train["label"])

    X_tr, y_tr = train[feats].values, le.transform(train["label"])
    X_te = test[feats].values
    y_te_labels = test["label"].values

    # pesos por classe: sem isso as classes raras somem
    counts = np.bincount(y_tr)
    weights = {i: len(y_tr) / (len(counts) * c) for i, c in enumerate(counts) if c}
    sample_weight = np.array([weights[y] for y in y_tr])

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(le.classes_),
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, sample_weight=sample_weight)

    proba = model.predict_proba(X_te)
    pred_labels = le.inverse_transform(proba.argmax(axis=1))

    known = np.isin(y_te_labels, le.classes_)
    report = classification_report(
        y_te_labels[known], pred_labels[known], zero_division=0, output_dict=True
    )
    fpr = false_positive_rate(y_te_labels, pred_labels)

    # PR-AUC binario: ataque vs benigno
    y_bin = (y_te_labels != BENIGN).astype(int)
    benign_idx = list(le.classes_).index(BENIGN) if BENIGN in le.classes_ else None
    score_attack = 1 - proba[:, benign_idx] if benign_idx is not None else proba.max(1)
    pr_auc = average_precision_score(y_bin, score_attack)

    print(classification_report(y_te_labels[known], pred_labels[known], zero_division=0))
    print(f"F1 macro : {report['macro avg']['f1-score']:.4f}")
    print(f"FPR      : {fpr:.5f}  ({fpr * 1_000_000:,.0f} alertas falsos por milhao de fluxos)")
    print(f"PR-AUC   : {pr_auc:.4f}")

    joblib.dump({"model": model, "label_encoder": le, "features": feats},
                args.models / "supervised.joblib")
    (args.models / "metrics_supervised.json").write_text(json.dumps(
        {"macro_f1": report["macro avg"]["f1-score"], "fpr": fpr, "pr_auc": pr_auc,
         "per_class": report}, indent=2))

    # SHAP: se uma unica feature domina tudo, provavelmente ha vazamento
    try:
        import shap
        sample = test[feats].sample(min(2000, len(test)), random_state=42)
        values = shap.TreeExplainer(model).shap_values(sample)
        imp = np.abs(np.array(values)).mean(axis=(0, 1)) if isinstance(values, list) \
            else np.abs(values).mean(axis=0)
        top = pd.Series(imp, index=feats).sort_values(ascending=False).head(15)
        print("\ntop 15 features (SHAP):")
        print(top.to_string())
    except ImportError:
        print("\n(shap nao instalado — pulando explicabilidade)")


if __name__ == "__main__":
    main()
