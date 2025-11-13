# Alternative Scoring Prototype (Console)

**Topic:** Alternative scoring for sole proprietors (ИП) without credit history: transactional data + behavioral metrics vs. classical scoring.

This console app trains two models on the PKDD'99 Czech financial dataset ("Berka"):
1) **Classical**: uses only static/bureau-like features (demographics & loan terms).
2) **Alternative**: adds **transactional** aggregates and **behavioral** metrics engineered from account transactions *prior to loan origination*.

It prints evaluation tables to the console and saves charts (ROC, PR) into `artifacts/`.

---

## 1) Get the data

Download the PKDD'99 (aka "Berka") CSV files and place them under `data/raw/` with the following names:

```
account.csv
card.csv
client.csv
disp.csv
district.csv
loan.csv
order.csv
trans.csv
```

Sources:
- Kaggle (Berka dataset mirror): see citations in the main answer.
- CTU relational repository (Financial / PKDD'99): see citations in the main answer.

> Tip: On Kaggle download the archive and extract the 8 CSV files. Keep the original column names.

---

## 2) Setup

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3) Run

### Train and evaluate both models
```bash
python -m src.main --data_dir data/raw --artifacts_dir artifacts --seed 42
```

### Show only quick EDA aggregates (no training)
```bash
python -m src.main --data_dir data/raw --artifacts_dir artifacts --seed 42 --mode eda
```

Outputs:
- Console tables with dataset stats and model metrics (AUC, F1, recall).
- Files saved in `artifacts/`:
  - `roc_classical.png`, `roc_alternative.png`
  - `pr_classical.png`, `pr_alternative.png`
  - `feature_importance_alternative.png`
  - `metrics.json`

---

## 4) Notes

- The code uses only **pandas**, **numpy**, **scikit-learn**, and **xgboost** (optional; falls back to RandomForest if xgboost not available).
- Transactional features are computed from a **lookback window** (default 180 days) before loan origination date per account.
- Behavioral metrics include: cashflow volatility, inflow/outflow ratios, overdraft frequency, balance volatility, and spending concentration.
- Feel free to modify `config.yaml` to tune windows and model hyperparams.
