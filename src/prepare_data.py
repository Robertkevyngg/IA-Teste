"""
Limpeza e preparacao do CIC-IDS2017 (versao corrigida).

Uso:
    python src/prepare_data.py --input data/raw --output data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Colunas que entregam a resposta de graca (vazamento) ou nao generalizam.
LEAKY_COLUMNS = [
    "Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
    "Source Port", "Src Port", "Timestamp",
]

# Colunas onde o CICFlowMeter produz Inf quando a duracao do fluxo e zero.
INF_COLUMNS = ["Flow Bytes/s", "Flow Packets/s"]


def load_raw(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"nenhum CSV encontrado em {input_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df["__source_file"] = f.name
        frames.append(df)
        print(f"  {f.name}: {len(df):,} linhas")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # 1. rotulos normalizados
    label_col = "Label" if "Label" in df.columns else df.columns[-2]
    df[label_col] = df[label_col].astype(str).str.strip()

    # 2. Inf -> NaN -> descarta
    for col in INF_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    numeric = df.select_dtypes(include=[np.number]).columns
    df = df.dropna(subset=numeric)
    print(f"  apos tratar Inf/NaN: {len(df):,} ({before - len(df):,} removidas)")

    # 3. duplicatas exatas — o CIC-IDS2017 tem muitas
    before = len(df)
    df = df.drop_duplicates()
    print(f"  apos duplicatas: {len(df):,} ({before - len(df):,} removidas)")

    # 4. colunas com vazamento e colunas constantes
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])
    constant = [c for c in df.select_dtypes(include=[np.number]).columns
                if df[c].nunique() <= 1]
    df = df.drop(columns=constant)
    print(f"  removidas {len(constant)} colunas constantes")

    return df.rename(columns={label_col: "label"})


def temporal_split(df: pd.DataFrame, test_size: float = 0.25):
    """
    Split POR ARQUIVO/DIA, nunca aleatorio.

    Fluxos de um mesmo ataque sao quase identicos entre si. Um split aleatorio
    coloca copias praticamente iguais no treino e no teste, e o modelo reporta
    99,9% que nao se sustenta em nenhum outro dado.
    """
    files = sorted(df["__source_file"].unique())
    n_test = max(1, int(round(len(files) * test_size)))
    test_files = files[-n_test:]
    train = df[~df["__source_file"].isin(test_files)].drop(columns="__source_file")
    test = df[df["__source_file"].isin(test_files)].drop(columns="__source_file")
    print(f"  treino: {len(train):,} | teste: {len(test):,} (arquivos {test_files})")
    return train, test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("data/processed"))
    p.add_argument("--test-size", type=float, default=0.25)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("carregando...")
    df = load_raw(args.input)
    print("limpando...")
    df = clean(df)
    print("dividindo (temporal)...")
    train, test = temporal_split(df, args.test_size)

    train.to_parquet(args.output / "train.parquet", index=False)
    test.to_parquet(args.output / "test.parquet", index=False)

    print("\ndistribuicao de classes no treino:")
    print(train["label"].value_counts().to_string())
    print(f"\nsalvo em {args.output}")


if __name__ == "__main__":
    main()
