"""
Limpeza e preparacao do CIC-IDS2017.

Uso:
    python src/prepare_data.py --input data/raw/archive --output data/processed
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Ordem cronologica real da captura. O nome do arquivo nao ordena por data
# (alfabeticamente "Tuesday" vem depois de "Monday" e "Thursday", o que
# inverteria a semana), entao a ordem e explicita.
DAY_ORDER = [
    "Monday-WorkingHours",
    "Tuesday-WorkingHours",
    "Wednesday-workingHours",
    "Thursday-WorkingHours-Morning-WebAttacks",
    "Thursday-WorkingHours-Afternoon-Infilteration",
    "Friday-WorkingHours-Morning",
    "Friday-WorkingHours-Afternoon-PortScan",
    "Friday-WorkingHours-Afternoon-DDos",
]

LEAKY_COLUMNS = [
    "Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
    "Source Port", "Src Port", "Timestamp",
]

INF_COLUMNS = ["Flow Bytes/s", "Flow Packets/s"]


def day_rank(filename: str) -> int:
    stem = filename.replace(".pcap_ISCX.csv", "").replace(".csv", "")
    for i, day in enumerate(DAY_ORDER):
        if stem.lower() == day.lower():
            return i
    return len(DAY_ORDER)  # arquivos desconhecidos vao para o fim


def load_raw(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"), key=lambda f: day_rank(f.name))
    if not files:
        raise FileNotFoundError(f"nenhum CSV encontrado em {input_dir}")
    frames = []
    for order, f in enumerate(files):
        # latin-1: os rotulos "Web Attack" usam um travessao cp1252 que quebra em utf-8
        df = pd.read_csv(f, low_memory=False, encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]
        df["__file"] = f.name
        df["__day"] = order
        df["__row"] = np.arange(len(df))  # ordem temporal dentro do dia
        frames.append(df)
        print(f"  [{order}] {f.name}: {len(df):,} linhas")
    return pd.concat(frames, ignore_index=True)


def normalize_label(value: str) -> str:
    """'Web Attack \\x96 Brute Force' -> 'Web Attack - Brute Force'."""
    text = re.sub(r"[^\x20-\x7E]+", "-", str(value)).strip()
    return re.sub(r"\s*-\s*", " - ", re.sub(r"\s+", " ", text)).strip()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    label_col = "Label" if "Label" in df.columns else df.columns[-4]
    df["label"] = df[label_col].map(normalize_label)
    if label_col != "label":
        df = df.drop(columns=[label_col])

    before = len(df)
    for col in INF_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if not c.startswith("__")]
    df = df.dropna(subset=feature_cols)
    print(f"  apos tratar Inf/NaN: {len(df):,} ({before - len(df):,} removidas)")

    before = len(df)
    df = df.drop_duplicates(subset=feature_cols + ["label"])
    print(f"  apos duplicatas: {len(df):,} ({before - len(df):,} removidas)")

    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])
    constant = [c for c in df.select_dtypes(include=[np.number]).columns
                if not c.startswith("__") and df[c].nunique() <= 1]
    df = df.drop(columns=constant)
    print(f"  removidas {len(constant)} colunas constantes")

    return df


def grouped_temporal_split(df: pd.DataFrame, test_size: float = 0.25):
    """
    Split temporal DENTRO de cada (dia, classe).

    Um split por dia inteiro nao serve no CIC-IDS2017: cada ataque acontece num
    unico dia, entao dias inteiros no teste deixam classes fora do treino.
    Um split aleatorio tambem nao serve: fluxos vizinhos de um mesmo ataque sao
    quase identicos e vazariam do treino para o teste.

    A solucao: ordenar por tempo dentro de cada (dia, classe) e cortar os
    primeiros (1 - test_size) para treino. Todas as classes aparecem dos dois
    lados, e nenhum fluxo de teste vem de antes do seu par de treino.
    """
    df = df.sort_values(["__day", "__row"], kind="stable")
    cut = df.groupby(["__day", "label"], sort=False).cumcount()
    sizes = df.groupby(["__day", "label"], sort=False)["label"].transform("size")
    is_test = cut >= np.ceil(sizes * (1 - test_size))

    drop = ["__file", "__day", "__row"]
    train = df[~is_test].drop(columns=drop)
    test = df[is_test].drop(columns=drop)
    print(f"  treino: {len(train):,} | teste: {len(test):,}")

    missing = set(train["label"]) ^ set(test["label"])
    if missing:
        print(f"  AVISO: classes presentes em so um dos lados: {sorted(missing)}")
    return train, test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("data/processed"))
    p.add_argument("--test-size", type=float, default=0.25)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("carregando (ordem cronologica)...")
    df = load_raw(args.input)
    print("limpando...")
    df = clean(df)
    print("dividindo (temporal por dia e classe)...")
    train, test = grouped_temporal_split(df, args.test_size)

    train.to_parquet(args.output / "train.parquet", index=False)
    test.to_parquet(args.output / "test.parquet", index=False)

    dist = pd.DataFrame({
        "treino": train["label"].value_counts(),
        "teste": test["label"].value_counts(),
    }).fillna(0).astype(int)
    dist["%_treino"] = (dist["treino"] / dist["treino"].sum() * 100).round(3)
    print("\ndistribuicao de classes:")
    print(dist.to_string())
    print(f"\nsalvo em {args.output}")


if __name__ == "__main__":
    main()
