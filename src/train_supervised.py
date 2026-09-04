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


def log(msg: str) -> None:
    print(msg, flush=True)


def load(processed: Path):
    log(f"carregando {processed / 'train.parquet'} ...")
    train = pd.read_parquet(processed / "train.parquet")
    log(f"  treino: {len(train):,} linhas")
    test = pd.read_parquet(processed / "test.parquet")
    log(f"  teste:  {len(test):,} linhas")
    feats = [c for c in train.columns if c != "label"]
    log(f"  features: {len(feats)}")
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
    p.add_argument("--n-estimators", type=int, default=400,
                   help="numero de arvores; use 100 para uma primeira passada rapida")
    p.add_argument("--sample", type=int, default=0,
                   help="treina em uma amostra de N linhas (0 = tudo)")
    args = p.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)

    import lightgbm as lgb

    train, test, feats = load(args.processed)

    if args.sample and args.sample < len(train):
        # amostra estratificada: mantem todas as classes, inclusive as raras
        frac = args.sample / len(train)
        # groupby().sample() preserva a coluna de agrupamento; groupby().apply()
        # a remove no pandas 2.2+, o que quebrava o LabelEncoder logo em seguida.
        train = train.groupby("label", group_keys=False).sample(
            frac=frac, random_state=42)
        log(f"amostra de treino: {len(train):,} linhas")

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
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    log(f"treinando LightGBM: {args.n_estimators} arvores, {len(le.classes_)} classes, "
        f"{X_tr.shape[0]:,} x {X_tr.shape[1]} ...")
    log("(o progresso aparece a cada 10 arvores; a primeira leva alguns segundos)")

    eval_n = min(50_000, len(X_te))
    model.fit(
        X_tr, y_tr, sample_weight=sample_weight,
        eval_set=[(X_te[:eval_n], le.transform(
            [l if l in le.classes_ else le.classes_[0] for l in y_te_labels[:eval_n]]))],
        eval_metric="multi_logloss",
        callbacks=[lgb.log_evaluation(period=10)],
    )
    log("treino concluido — avaliando...")

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
        values = np.abs(np.array(shap.TreeExplainer(model).shap_values(sample)))
        # multiclasse devolve 3 dimensoes; a ordem dos eixos muda entre versoes
        # do shap, entao localizamos o eixo das features pelo tamanho.
        if values.ndim == 3:
            axis = next(i for i, n in enumerate(values.shape) if n == len(feats))
            imp = values.mean(axis=tuple(i for i in range(3) if i != axis))
        else:
            imp = values.mean(axis=0)
        top = pd.Series(imp, index=feats).sort_values(ascending=False).head(15)
        print("\ntop 15 features (SHAP):")
        print(top.to_string())
    except ImportError:
        print("\n(shap nao instalado — pulando explicabilidade)")


if __name__ == "__main__":
    main()
