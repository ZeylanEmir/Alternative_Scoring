import pandas as pd
import numpy as np
from .behavior import build_behavior_features

def _read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    df.columns = df.columns.str.strip().str.lower()
    return df

def prepare_base_tables(data_dir: str):
    account = _read(f"{data_dir}/account.csv")
    client  = _read(f"{data_dir}/client.csv")
    disp    = _read(f"{data_dir}/disp.csv")
    loan    = _read(f"{data_dir}/loan.csv")
    trans   = _read(f"{data_dir}/trans.csv")
    district= _read(f"{data_dir}/district.csv")

    for df in [account, client, disp, loan, trans, district]:
        df.columns = [c.lower() for c in df.columns]

    # даты (Berka): дни от 1993-01-01
    for df in [loan, trans]:
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_timedelta(df['date'], unit='D') + pd.Timestamp('1993-01-01')
            except Exception:
                df['date'] = pd.to_datetime(df['date'])

    loans = loan.merge(account[['account_id','district_id']], on='account_id', how='left')
    owner = disp[disp['type'].str.lower().eq('owner')].merge(client, on='client_id', how='left')
    loans = loans.merge(owner[['account_id','client_id','birth_number','district_id']], on='account_id', how='left', suffixes=('','_owner'))

    if 'district_id' in district.columns:
        loans = loans.merge(district.add_prefix('dist_'), left_on='district_id', right_on='dist_district_id', how='left')

    return loans, trans

def _safe_div(a, b):
    return (a / b) if (b not in (0, 0.0, None) and not pd.isna(b)) else 0.0

def engineer_transactional_features(loans_df: pd.DataFrame, trans_df: pd.DataFrame, lookback_days: int = 180, min_txns: int = 5):
    feats = []
    need_cols = {'account_id','date','amount'}
    if not need_cols.issubset(set(trans_df.columns)):
        raise ValueError(f"Transaction table must contain columns {need_cols}")

    trans = trans_df.copy()
    if (trans['amount'] < 0).sum() == 0 and 'type' in trans.columns:
        trans['signed_amount'] = np.where(trans['type'].str.contains('PRIJ', case=False, na=False), trans['amount'], -trans['amount'])
    else:
        trans['signed_amount'] = trans['amount']

    trans = trans.sort_values(['account_id','date'])
    for _, row in loans_df[['loan_id','account_id','date']].dropna().iterrows():
        loan_id = row['loan_id']; acc = row['account_id']
        t_end = row['date']; t_start = t_end - pd.Timedelta(days=lookback_days)
        tx = trans[(trans['account_id']==acc) & (trans['date']>=t_start) & (trans['date']<t_end)]
        n = len(tx)
        if n == 0:
            feats.append(dict(loan_id=loan_id, n_txn=0))
            continue

        inflow = tx.loc[tx['signed_amount']>0, 'signed_amount'].sum()
        outflow = -tx.loc[tx['signed_amount']<0, 'signed_amount'].sum()
        net = inflow - outflow
        mean_in = tx.loc[tx['signed_amount']>0, 'signed_amount'].mean() if (tx['signed_amount']>0).any() else 0.0
        mean_out= -tx.loc[tx['signed_amount']<0, 'signed_amount'].mean() if (tx['signed_amount']<0).any() else 0.0

        daily = tx.set_index('date')['signed_amount'].resample('D').sum().reindex(pd.date_range(t_start, t_end - pd.Timedelta(days=1)), fill_value=0.0)
        vol = float(daily.std()); zero_days = float((daily == 0).mean())

        bal_vol = np.nan; overdrafts = 0
        if 'balance' in tx.columns:
            try:
                bal_daily = tx.set_index('date')['balance'].resample('D').last().ffill()
                bal_vol = float(bal_daily.std())
                overdrafts = int((tx['balance'] < 0).sum())
            except Exception:
                pass

        conc = np.nan
        if 'counterparty' in tx.columns:
            grp = tx.assign(absamt=tx['signed_amount'].abs()).groupby('counterparty', dropna=False)['absamt'].sum()
            if grp.sum() > 0 and len(grp) > 1:
                s = np.sort(grp.values);
                n_s = len(s)
                cum = np.cumsum(s) / s.sum()
                conc = float((n_s + 1 - 2 * cum.sum()) / n_s)
            else:
                conc = 0.0
        else:
            conc = 0.0  # <== ДОБАВЬ ЭТУ СТРОКУ (дефолт, если counterparty нет)

        feats.append(dict(
            loan_id=loan_id,
            n_txn=int(n),
            inflow=float(inflow),
            outflow=float(outflow),
            net=float(net),
            mean_in=float(mean_in),
            mean_out=float(mean_out),
            txn_per_day=n / max(1, (t_end - t_start).days),
            cashflow_vol=vol,
            inactivity_ratio=zero_days,
            balance_vol=bal_vol,
            overdraft_events=int(overdrafts),
            spend_concentration=conc,
            io_ratio=_safe_div(inflow, outflow) if outflow > 0 else np.nan,
        ))

    return pd.DataFrame(feats)

def classical_features(loans_df: pd.DataFrame):
    cols = [c for c in ['amount','duration','payments'] if c in loans_df.columns]
    return loans_df[cols + ['loan_id']].copy()

# === Шаг 4: интеграция поведения в обучение ===
def engineer_behavioral_window_features(loans_df: pd.DataFrame, trans_df: pd.DataFrame, lookback_days: int = 180):
    rows = []
    trans = trans_df.copy()
    if 'date' in trans.columns and not pd.api.types.is_datetime64_any_dtype(trans['date']):
        trans['date'] = pd.to_datetime(trans['date'], errors="coerce")
    for _, row in loans_df[['loan_id','account_id','date']].dropna().iterrows():
        loan_id = row['loan_id']; acc = row['account_id']
        t_end = row['date']; t_start = t_end - pd.Timedelta(days=lookback_days)
        tx = trans[(trans['account_id']==acc) & (trans['date']>=t_start) & (trans['date']<t_end)].copy()
        if tx.empty:
            rows.append({'loan_id': loan_id})  # пустая строка — пусть имьютер добьёт
            continue
        bf = build_behavior_features(tx)
        bf.insert(0, 'loan_id', loan_id)
        rows.append(bf.iloc[0].to_dict())
    return pd.DataFrame(rows)
