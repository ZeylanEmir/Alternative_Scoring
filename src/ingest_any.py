import pandas as pd, json, pathlib
import numpy as np
import os
from typing import Tuple
from .pdf_bank import pdf_to_transactions

REQUIRED_TRANS_COLS = ["date","amount","type","balance","counterparty"]

def read_any(path: str) -> Tuple[str, pd.DataFrame]:
    p = pathlib.Path(path)
    ext = p.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        kind = "table"
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        kind = "table"
    elif ext == ".json":
        obj = json.load(open(path, "r", encoding="utf-8"))
        df = pd.json_normalize(obj)
        kind = "table"
    elif ext == ".pdf":
        df = pdf_to_transactions(path)   # -> DataFrame с ["date","amount","type","balance","counterparty"]
        kind = "transactions"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Нормализация
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Если это табличный файл, но по факту транзакции — переключаем вид
    if kind == "table" and {"date","amount"}.issubset(set(df.columns)):
        kind = "transactions"
        # amount -> float (учёт скобок, запятых)
        df["amount"] = (
            df["amount"].astype(str)
            .str.replace(r"[^\d\-\.\,()]", "", regex=True)
            .str.replace("(", "-", regex=False).str.replace(")", "", regex=False)
        )
        df["amount"] = pd.to_numeric(df["amount"].str.replace(",", ".", regex=False), errors="coerce")
        # date -> datetime (разрешим day-first)
        if not np.issubdtype(df["date"].dtype, np.datetime64):
            df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        # type по знаку, если отсутствует
        if "type" not in df.columns:
            df["type"] = np.where(df["amount"] > 0, "PRIJEM", "VYDAJ")
        # баланс/контрагент — пусть будут, даже если NaN
        if "balance" not in df.columns:
            df["balance"] = np.nan
        if "counterparty" not in df.columns:
            # пробуем из company/merchant/memo
            for alt in ("company","merchant","memo","description"):
                if alt in df.columns:
                    df["counterparty"] = df[alt]
                    break
            else:
                df["counterparty"] = np.nan

    # Диагностика: превью
    os.makedirs("artifacts", exist_ok=True)
    preview_path = "artifacts/predict_preview.csv"
    df.head(25).to_csv(preview_path, index=False)
    print(f"[ingest] kind={kind}, rows={len(df)}, cols={list(df.columns)} -> preview: {preview_path}")

    return kind, df
