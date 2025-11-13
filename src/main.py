import argparse, json, sys, os
import pandas as pd
import numpy as np
from tabulate import tabulate
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from joblib import dump, load

from .features import (
    prepare_base_tables,
    engineer_transactional_features,
    classical_features,
    engineer_behavioral_window_features,   # NEW: шаг 4
)

from .behavior import build_behavior_features
from .ingest_any import read_any
from .scoring import pd_to_score
from .gauge import save_gauge

# === Optional: XGBoost if available ===
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def build_model(model_cfg: dict):
    mtype = model_cfg.get("type", "xgboost")
    if mtype == "xgboost" and HAS_XGB:
        params = model_cfg.get("xgb", {})
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.08),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            n_jobs=4,
            eval_metric="logloss",
            tree_method="hist",
            random_state=params.get("random_state", 42),
            # scale_pos_weight можно включить при сильном дисбалансе
        )
    params = model_cfg.get("rf", {})
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 400),
        max_depth=params.get("max_depth", 10),
        n_jobs=4,
        random_state=42,
    )

def train_and_eval(df: pd.DataFrame, target_col: str, model_cfg: dict, artifacts_dir: str, prefix: str):
    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ]
    )

    model = build_model(model_cfg)
    pipe = Pipeline([("pre", pre), ("clf", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.get("test_size", 0.25), random_state=cfg.get("random_state", 42), stratify=y
    )

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["clf"], "predict_proba") else pipe.decision_function(X_test)
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    ap  = average_precision_score(y_test, proba)
    f1  = f1_score(y_test, preds)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{prefix} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {prefix}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, f"roc_{prefix}.png")); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_test, proba)
    plt.figure()
    plt.plot(rec, prec, label=f"{prefix} AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {prefix}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, f"pr_{prefix}.png")); plt.close()

    return {"auc": float(auc), "ap": float(ap), "f1": float(f1)}, pipe, X.columns.tolist()

def infer_target(loans: pd.DataFrame):
    st = loans.get("status")
    if st is None:
        raise ValueError("Loan table must include a 'status' column for target inference.")
    st = st.astype(str).str.upper().str.strip()
    return (st == "D").astype(int).rename("target")

# === Фолбэк агрегаций из «сырых» транзакций (для predict) ===
def _fallback_tx_features_from_raw(tx: pd.DataFrame) -> pd.DataFrame:
    df = tx.copy()
    if "amount" not in df.columns:
        raise ValueError("Transactions table must contain 'amount' column.")
    df["signed_amount"] = pd.to_numeric(df["amount"], errors="coerce")

    inflow = float(df.loc[df["signed_amount"]>0, "signed_amount"].sum())
    outflow= float(-df.loc[df["signed_amount"]<0, "signed_amount"].sum())
    net = inflow - outflow
    mean_in  = float(df.loc[df["signed_amount"]>0, "signed_amount"].mean() or 0.0)
    mean_out = float(-df.loc[df["signed_amount"]<0, "signed_amount"].mean() or 0.0)

    vol = np.nan; inact = np.nan
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            d0, d1 = df["date"].min().normalize(), df["date"].max().normalize()
            daily = df.set_index("date")["signed_amount"].resample("D").sum().reindex(pd.date_range(d0, d1), fill_value=0.0)
            vol = float(daily.std()); inact = float((daily == 0).mean())
        except Exception:
            pass

    overdrafts = 0; bal_vol = np.nan
    if "balance" in df.columns and df["balance"].notna().any():
        try:
            bal_daily = df.set_index("date")["balance"].resample("D").last().ffill()
            bal_vol = float(bal_daily.std())
            overdrafts = int((df["balance"] < 0).sum())
        except Exception:
            pass

    conc = 0.0
    if "counterparty" in df.columns:
        grp = df.assign(absamt=df["signed_amount"].abs()).groupby("counterparty", dropna=False)["absamt"].sum()
        if grp.sum() > 0 and len(grp) > 1:
            s = np.sort(grp.values);
            n = len(s)
            cum = np.cumsum(s) / s.sum()
            conc = float((n + 1 - 2 * cum.sum()) / n)

    io_ratio = inflow / outflow if outflow > 0 else np.nan

    return pd.DataFrame([dict(
        n_txn=int(len(df)),
        inflow=inflow,
        outflow=outflow,
        net=net,
        mean_in=mean_in,
        mean_out=mean_out,
        txn_per_day=np.nan,
        cashflow_vol=float(vol) if not pd.isna(vol) else np.nan,
        inactivity_ratio=float(inact) if not pd.isna(inact) else np.nan,
        balance_vol=float(bal_vol) if not pd.isna(bal_vol) else np.nan,
        overdraft_events=int(overdrafts),
        spend_concentration=float(conc) if not pd.isna(conc) else np.nan,
        io_ratio=float(io_ratio) if not pd.isna(io_ratio) else np.nan,
    )])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=False, default="data/raw")
    p.add_argument("--artifacts_dir", type=str, default="artifacts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--mode", choices=["train", "predict", "eda"], default="train")
    p.add_argument("--input_file", type=str, default=None, help="CSV/XLSX/JSON/PDF for predict")
    p.add_argument("--gauge_png", type=str, default="artifacts/credit_gauge.png")
    args = p.parse_args()

    global cfg
    cfg = dict(
        lookback_days=180,
        min_txns=5,
        test_size=0.25,
        random_state=args.seed,
        model={
            "type": "xgboost",
            "xgb": {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.08,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": args.seed,
            },
            "rf": {"n_estimators": 400, "max_depth": 10},
        },
    )
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f)
            cfg.update(user_cfg or {})

    os.makedirs(args.artifacts_dir, exist_ok=True)

    # ---------- PREDICT ----------
    if args.mode == "predict":
        if not args.input_file:
            raise ValueError("--input_file is required for predict.")
        class_pipe_path = os.path.join(args.artifacts_dir, "class_pipeline.joblib")
        alt_pipe_path   = os.path.join(args.artifacts_dir, "alt_pipeline.joblib")
        class_meta_path = os.path.join(args.artifacts_dir, "class_features.json")
        alt_meta_path   = os.path.join(args.artifacts_dir, "alt_features.json")
        for pth in [class_pipe_path, alt_pipe_path, class_meta_path, alt_meta_path]:
            if not os.path.exists(pth):
                raise FileNotFoundError(f"Missing artifact: {pth}. Run train mode first.")

        class_pipe = load(class_pipe_path)
        alt_pipe   = load(alt_pipe_path)
        class_cols = json.loads(open(class_meta_path, "r", encoding="utf-8").read())["feature_columns"]
        alt_cols   = json.loads(open(alt_meta_path, "r", encoding="utf-8").read())["feature_columns"]

        kind, df_in = read_any(args.input_file)
        if df_in is None or len(df_in) == 0:
            raise ValueError("Input file parsed, but no rows were extracted. Check artifacts/predict_preview.csv")

        # Если по факту транзакции — так и считаем
        if kind == "table" and {"date","amount"}.issubset(set(map(str.lower, df_in.columns))):
            kind = "transactions"

        print(f"[predict] received kind={kind}, shape={df_in.shape}")

        # CLASSIC фичи (здесь как есть, чаще всего NaN, это ок — демонстрационный трек)
        Xc = pd.DataFrame(columns=class_cols)
        for c in class_cols:
            Xc[c] = df_in[c] if c in df_in.columns else np.nan

        # ALT фичи: агрегаты + поведение
        if kind == "transactions":
            tx_feat = _fallback_tx_features_from_raw(df_in)
            Xa = pd.DataFrame(columns=alt_cols)
            for c in alt_cols:
                if c in tx_feat.columns:
                    Xa[c] = tx_feat[c]
                elif c in df_in.columns:
                    Xa[c] = df_in[c]
                else:
                    Xa[c] = np.nan
            # поведение (шаг 5 — и для модели, и для объяснимости)
            bf = build_behavior_features(df_in)
            # добавим то, что модель знает
            for c in bf.columns:
                if c in Xa.columns:
                    Xa[c] = bf[c]
        else:
            Xa = pd.DataFrame(columns=alt_cols)
            for c in alt_cols:
                Xa[c] = df_in[c] if c in df_in.columns else np.nan
            bf = pd.DataFrame([{}])  # для explain блока ниже

        # proba helper
        def _proba(pipe, X):
            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                return pipe.predict_proba(X)[:, 1]
            raw = pipe.decision_function(X)
            return 1 / (1 + np.exp(-raw))

        pd_class = float(np.clip(_proba(class_pipe, Xc)[0], 1e-9, 1 - 1e-9))
        pd_alt   = float(np.clip(_proba(alt_pipe, Xa)[0],   1e-9, 1 - 1e-9))
        score_class = int(pd_to_score(pd_class, min_score=1, max_score=1000, base_points=600.0, base_odds=20.0, pdo=50.0))
        score_alt   = int(pd_to_score(pd_alt,   min_score=1, max_score=1000, base_points=600.0, base_odds=20.0, pdo=50.0))

        print("\n=== Prediction (single-row) ===")
        rows = [
            {"track": "CLASSIC", "PD": pd_class, "Score(1..1000)": score_class},
            {"track": "ALT",     "PD": pd_alt,   "Score(1..1000)": score_alt},
            {"track": "DELTA",   "PD": pd_alt - pd_class, "Score(1..1000)": score_alt - score_class},
        ]
        print(tabulate(rows, headers="keys", tablefmt="github", floatfmt=".6f"))

        # === Шаг 5. Объяснимость ===
        explain = {}
        if not bf.empty:
            row = bf.iloc[0].to_dict()
            explain = {
                "top3": [
                    {"cat": row.get("top1_cat","-"), "share": float(row.get("top1_share", 0))},
                    {"cat": row.get("top2_cat","-"), "share": float(row.get("top2_share", 0))},
                    {"cat": row.get("top3_cat","-"), "share": float(row.get("top3_share", 0))},
                ],
                "risky_share": float(row.get("risky_share", 0)),
                "p2p_ratio": float(row.get("p2p_ratio", 0)),
                "cash_ratio": float(row.get("cash_ratio", 0)),
                "merchant_concentration": float(row.get("merchant_concentration", 0)),
            }

            notes = []
            if explain["risky_share"] > 0.20: notes.append("High risky share (>20%): betting/crypto -> higher risk.")
            if explain["cash_ratio"]  > 0.30: notes.append("High cash ratio (>30%): low transparency -> higher risk.")
            if explain["p2p_ratio"]   > 0.50: notes.append("Dominant P2P (>50%): unstable income -> higher risk.")
            if explain["merchant_concentration"] > 0.60: notes.append("High concentration (>0.6): dependency on few counterparties.")
            explain["notes"] = notes

            print("\n=== Behavioral Explainability ===")
            print(tabulate([
                ["Top-1", f"{explain['top3'][0]['cat']} ({explain['top3'][0]['share']:.2f})"],
                ["Top-2", f"{explain['top3'][1]['cat']} ({explain['top3'][1]['share']:.2f})"],
                ["Top-3", f"{explain['top3'][2]['cat']} ({explain['top3'][2]['share']:.2f})"],
                ["risky_share", f"{explain['risky_share']:.2f}"],
                ["p2p_ratio", f"{explain['p2p_ratio']:.2f}"],
                ["cash_ratio", f"{explain['cash_ratio']:.2f}"],
                ["merchant_concentration", f"{explain['merchant_concentration']:.2f}"],
                ["notes", "; ".join(notes) if notes else "—"],
            ], headers=["metric","value"], tablefmt="github"))

        # save json + gauge
        out_json = os.path.join(args.artifacts_dir, "predictions.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "pd_class": pd_class, "score_class": score_class,
                "pd_alt": pd_alt, "score_alt": score_alt,
                "explain": explain
            }, f, indent=2)
        save_gauge(score_alt, min_s=1, max_s=1000, path=args.gauge_png, title="Alt Credit Score")

        sys.exit(0)

    # ---------- TRAIN / EDA ----------
    loans, trans = prepare_base_tables(args.data_dir)
    target = infer_target(loans)
    loans = loans.merge(target, left_index=True, right_index=True)

    eda_tbl = loans["target"].value_counts().rename({0: "non-default", 1: "default"}).to_frame("count")
    eda_tbl["ratio"] = eda_tbl["count"] / eda_tbl["count"].sum()
    print("\n=== Target Distribution ===")
    print(tabulate(eda_tbl.reset_index().rename(columns={"index": "class"}), headers="keys", tablefmt="github", showindex=False))

    if args.mode == "eda":
        print("\n=== Loans sample ===")
        print(tabulate(loans.head(10), headers="keys", tablefmt="github", showindex=False))
        print("\n=== Transactions sample ===")
        print(tabulate(trans.head(10), headers="keys", tablefmt="github", showindex=False))
        sys.exit(0)

    # Классический трек
    X_classical = classical_features(loans).copy()
    X_classical = X_classical.merge(loans[["loan_id","target"]], on="loan_id", how="left").dropna(subset=["target"])

    # Альтернативный трек: транзакционные агрегаты + поведение (Шаг 4)
    tx_feat = engineer_transactional_features(loans, trans, lookback_days=cfg.get("lookback_days", 180), min_txns=cfg.get("min_txns", 5))
    bh_feat = engineer_behavioral_window_features(loans, trans, lookback_days=cfg.get("lookback_days", 180))
    X_alt = loans[["loan_id","amount","duration","payments","target"]].merge(tx_feat, on="loan_id", how="left").merge(bh_feat, on="loan_id", how="left")

    all_nan_cols = [c for c in X_alt.columns if X_alt[c].isna().all()]
    if all_nan_cols:
        X_alt = X_alt.drop(columns=all_nan_cols)

    if "balance_vol" in X_alt.columns and X_alt["balance_vol"].isna().all():
        X_alt = X_alt.drop(columns=["balance_vol"])

    X_alt = X_alt.dropna(subset=["target"])

    print("\n=== Training Classical Model (no transactions) ===")
    m_class, pipe_class, cols_class = train_and_eval(
        X_classical.drop(columns=["loan_id"]), "target", cfg["model"], args.artifacts_dir, prefix="classical"
    )
    print(tabulate(pd.DataFrame([m_class]), headers="keys", tablefmt="github", showindex=False))

    print("\n=== Training Alternative Model (transactions + behavioral) ===")
    m_alt, pipe_alt, cols_alt = train_and_eval(
        X_alt.drop(columns=["loan_id"]), "target", cfg["model"], args.artifacts_dir, prefix="alternative"
    )
    print(tabulate(pd.DataFrame([m_alt]), headers="keys", tablefmt="github", showindex=False))

    dump(pipe_class, os.path.join(args.artifacts_dir, "class_pipeline.joblib"))
    dump(pipe_alt,   os.path.join(args.artifacts_dir, "alt_pipeline.joblib"))
    with open(os.path.join(args.artifacts_dir, "class_features.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_columns": cols_class}, f, indent=2)
    with open(os.path.join(args.artifacts_dir, "alt_features.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_columns": cols_alt}, f, indent=2)
    with open(os.path.join(args.artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"classical": m_class, "alternative": m_alt}, f, indent=2)

    print(f"\nArtifacts saved to: {args.artifacts_dir}")
