import pandas as pd
import numpy as np
import re

# Ключевые «риск-» маркеры по названиям
RISKY_PATTERNS = {
    "betting":  r"(?:1xbet|bet365|mostbet|parimatch|pari match|букмек|casino|казино)",
    "crypto":   r"(?:binance|bybit|okx|kucoin|coinbase|kraken|crypto)",
    "adult":    r"(?:onlyfans|pornhub|brazzers|fansly)",
}
P2P_PAT   = r"(?:p2p|p2p transfer|p2p перевод|kaspi p2p|p2p payment)"
CASH_PAT  = r"(?:atm|cash|налич|банкомат|atm cash)"
INCOME_PAT= r"(?:salary|зарплат|доход|revenue|выручка|поступлен|marketplace)"
def _ratio(x: float, y: float) -> float:
    return float(x) / float(y) if (y is not None and y != 0) else 0.0

def _safe_sum(s):
    return float(pd.to_numeric(s, errors="coerce").fillna(0).sum())

def _mask_from_counterparty(series: pd.Series, pattern: str):
    # важно: na=False, чтобы NaN не давали предупреждений и считались False
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
    )
def build_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Возвращает одну строку с поведенческими фичами."""
    d = df.copy()
    # гарантируем нужные поля
    for col in ("amount","type","counterparty","mcc","date"):
        if col not in d.columns:
            d[col] = np.nan
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    d["abs"] = d["amount"].abs()
    d["is_out"] = d["amount"] < 0
    d["is_in"]  = d["amount"] > 0
    if not pd.api.types.is_datetime64_any_dtype(d["date"]):
        d["date"] = pd.to_datetime(d["date"], errors="coerce")

    total_abs = _safe_sum(d["abs"])
    total_out = _safe_sum(d.loc[d["is_out"], "abs"])
    total_in  = _safe_sum(d.loc[d["is_in"],  "abs"])
    n_txn     = int(len(d))

    # Базовые доли
    cash_mask = _mask_from_counterparty(d["counterparty"], CASH_PAT) | _mask_from_counterparty(d.get("description", d["counterparty"]), CASH_PAT)
    p2p_mask  = _mask_from_counterparty(d["counterparty"], P2P_PAT)
    inc_mask  = _mask_from_counterparty(d["counterparty"], INCOME_PAT)

    # Рисковые категории
    risk_masks = {}
    for k, pat in RISKY_PATTERNS.items():
        risk_masks[k] = _mask_from_counterparty(d["counterparty"], pat)

    risky_sum = 0.0
    for k in risk_masks:
        risky_sum += _safe_sum(d.loc[risk_masks[k], "abs"])

    # Концентрация торговцев (джинни по контрагентам по абсолютному обороту)
    by_cp = d.groupby(d["counterparty"].fillna("NA"))["abs"].sum()
    if by_cp.sum() > 0 and len(by_cp) > 1:
        s = np.sort(by_cp.values)
        n = len(s)
        cum = np.cumsum(s) / s.sum()
        concentration = float((n + 1 - 2 * cum.sum()) / n)
    else:
        concentration = 0.0

    # Периоды без движения (инертность)
    inactivity_ratio = np.nan
    if d["date"].notna().any():
        try:
            d0, d1 = d["date"].min().normalize(), d["date"].max().normalize()
            daily = d.set_index("date")["amount"].resample("D").sum().reindex(pd.date_range(d0, d1), fill_value=0.0)
            inactivity_ratio = float((daily == 0).mean())
        except Exception:
            inactivity_ratio = np.nan

    # Сбор итогов
    feat = dict(
        n_txn = n_txn,
        turnover_abs = total_abs,
        turnover_out = total_out,
        turnover_in  = total_in,
        cash_ratio = _ratio(_safe_sum(d.loc[cash_mask, "abs"]), total_abs),
        p2p_ratio  = _ratio(_safe_sum(d.loc[p2p_mask  , "abs"]), total_abs),
        income_ratio = _ratio(_safe_sum(d.loc[inc_mask , "abs"]), total_abs),
        risky_share = _ratio(risky_sum, total_abs),
        risky_betting_share = _ratio(_safe_sum(d.loc[risk_masks["betting"], "abs"]), total_abs),
        risky_crypto_share  = _ratio(_safe_sum(d.loc[risk_masks["crypto"],  "abs"]), total_abs),
        merchant_concentration = float(concentration),
        inactivity_ratio = float(inactivity_ratio) if not pd.isna(inactivity_ratio) else np.nan,
    )

    # Топ-3 категорий по доле (для explain в predict)
    cats = {
        "betting": feat["risky_betting_share"],
        "crypto": feat["risky_crypto_share"],
        "p2p": feat["p2p_ratio"],
        "cash": feat["cash_ratio"],
        "income": feat["income_ratio"],
        "other": max(0.0, 1.0 - (feat["risky_betting_share"]+feat["risky_crypto_share"]+feat["p2p_ratio"]+feat["cash_ratio"]+feat["income_ratio"])),
    }
    top3 = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:3]
    for i, (name, val) in enumerate(top3, 1):
        feat[f"top{i}_cat"] = name
        feat[f"top{i}_share"] = float(val)

    return pd.DataFrame([feat])
