# src/pdf_bank.py
import re
from datetime import datetime
import pandas as pd
import pdfplumber
import numpy as np
import re

_DATE_PATTERNS = ("%d/%m/%Y","%m/%d/%Y","%d.%m.%Y","%Y-%m-%d","%d %b %Y","%b %d %Y")
_CURRENCY_CLEAN = re.compile(r"[^\d\-\.\,\(\)]")

def _try_parse_date(s: str):
    if s is None: return None
    t = str(s).strip()
    for fmt in _DATE_PATTERNS:
        try: return datetime.strptime(t, fmt)
        except: pass
    return None

def _to_float(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    s = str(x).strip()
    if not s: return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True; s = s[1:-1]
    s = _CURRENCY_CLEAN.sub("", s)
    # десятичная запятая
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    # разделители тысяч
    if s.count(",") > 0 and s.count(".") > 0:
        s = s.replace(",", "")
    try:
        v = float(s)
        return -v if neg else v
    except: return None

def _norm_header(h):  # нормализация заголовков
    return str(h or "").strip().lower().replace("\n"," ").replace("\r"," ")

HEADER_MAP = {
    "date": ["date","transaction date","posted","book date","datum","дата"],
    "desc": ["description","details","narration","memo","payee","beneficiary","описание","назначение"],
    # эти имена важны для твоего PDF:
    "debit": ["debit","withdrawal","withdrawals","paid out","payment","vydej","debits","списание","debet","withdrawals ($)"],
    "credit":["credit","deposit","deposits","paid in","receipt","prijem","credits","зачисление","kredit","deposits ($)"],
    "balance":["balance","balance ($)","closing balance","ending balance","zůstatek","остаток"],
    "amount": ["amount","amt","sum","сумма","сумма операции"]
}

def _key_by_header(hdr: list[str]) -> dict:
    normed = [_norm_header(x) for x in hdr]
    mapping = {}
    for std, aliases in HEADER_MAP.items():
        found = None
        for i, h in enumerate(normed):
            if h in aliases: found = hdr[i]; break
        if not found:
            for i, h in enumerate(normed):
                if any(a in h for a in aliases):
                    found = hdr[i]; break
        if found: mapping[std] = found
    return mapping

def _first_present(d: dict, key):
    if key and key in d and d[key] not in (None,"","nan"): return d[key]
    return None

def _build_amount(row, m):
    if "amount" in m:
        v = _to_float(row.get(m["amount"]))
        if v is not None: return v
    debit = _to_float(row.get(m.get("debit"))) if "debit" in m else None
    credit= _to_float(row.get(m.get("credit"))) if "credit" in m else None
    if debit is not None and abs(debit) > 0:  return -abs(debit)
    if credit is not None and abs(credit) > 0: return  abs(credit)
    return None

def _collect_rows_from_matrix(tbl_matrix):
    rows = []
    if not tbl_matrix or len(tbl_matrix) <= 1: return rows
    hdr = [c if c is not None else "" for c in tbl_matrix[0]]
    mapping = _key_by_header(hdr)
    for r in tbl_matrix[1:]:
        rec = {hdr[i]: r[i] for i in range(min(len(hdr), len(r)))}
        date = _first_present(rec, mapping.get("date")) if "date" in mapping else None
        desc = _first_present(rec, mapping.get("desc")) if "desc" in mapping else None
        bal  = _first_present(rec, mapping.get("balance")) if "balance" in mapping else None
        amt  = _build_amount(rec, mapping)
        dt = _try_parse_date(date)
        if dt and amt is not None:
            rows.append({
                "date": dt,
                "amount": amt,
                "type": "PRIJEM" if amt > 0 else "VYDAJ",
                "balance": _to_float(bal) if bal is not None else None,
                "counterparty": desc
            })
    return rows

_MON_MAP = {m.lower(): i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}

def _words_to_rows(page):
    """Группируем слова по строкам (y-биннинг). Возвращаем список: [ {y0,y1, words:[(x0,x1,text), ...]} ]"""
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    if not words: return []
    lines = []
    # бинируем по top/bottom
    for w in words:
        x0, x1 = float(w["x0"]), float(w["x1"])
        top, bottom = float(w["top"]), float(w["bottom"])
        txt = w["text"]
        placed = False
        for L in lines:
            # если слово попадает в «полосу» по y — прикрепляем
            if not (bottom < L["y0"] or top > L["y1"]):
                L["y0"] = min(L["y0"], top)
                L["y1"] = max(L["y1"], bottom)
                L["words"].append((x0, x1, txt))
                placed = True
                break
        if not placed:
            lines.append({"y0": top, "y1": bottom, "words": [(x0, x1, txt)]})
    # упорядочим по вертикали и по x внутри строки
    lines.sort(key=lambda z: z["y0"])
    for L in lines:
        L["words"].sort(key=lambda t: t[0])
    return lines

_CLEAN_NUM = re.compile(r"[^\d\-\.\,\(\)]")
def _to_amount(txt):
    if txt is None: return None
    s = str(txt).strip()
    neg = False
    if s.startswith("(") and s.endswith(")"): neg = True; s = s[1:-1]
    s = _CLEAN_NUM.sub("", s)
    # десятичная запятая
    if s.count(",")==1 and s.count(".")==0:
        s = s.replace(",", ".")
    # тысячные пробелы/запятые
    s = s.replace(" ", "").replace(",", "") if s.count(".")==1 else s.replace(" ", "")
    try:
        v = float(s)
        return -v if neg else v
    except:
        return None

def _parse_text_fallback(path):
    """Пробуем распарсить:
       A) Kaspi-подобный формат: 'Date Amount Transaction Details', суммы со знаком и '₸'
       B) Withdrawals/Deposits формат (CIBC): определяем границы столбцов по x заголовков.
    """
    rows = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            lines = _words_to_rows(page)
            # собрать текст заголовка (по первой «широкой» строке с ключевыми словами)
            header_idx = None
            header_txt = ""
            for i, L in enumerate(lines[:15]):
                t = " ".join(w[2] for w in L["words"]).lower()
                if any(k in t for k in ["date", "datum", "дата"]):
                    header_idx = i; header_txt = t; break

            if header_idx is None:
                # не нашли заголовок — пропустим страницу
                continue

            # === ВЕТКА Kaspi: есть "amount" и "transaction" ===
            if "amount" in header_txt and ("transaction" in header_txt or "details" in header_txt):
                for L in lines[header_idx+1:]:
                    t = " ".join(w[2] for w in L["words"])
                    # дата dd.mm.yy в начале строки
                    m = re.match(r"^\s*(\d{2}\.\d{2}\.\d{2})\s+(.*?)$", t)
                    if not m:
                        continue
                    dstr, rest = m.group(1), m.group(2)
                    # сумма со знаком, перед символом валюты (₸, $, т.д.)
                    m2 = re.search(r"([+\-]?\s*[\d\s\.,\(\)]+)\s*[₸$€]|([+\-]\s*[\d\s\.,\(\)]+)$", rest)
                    if not m2:
                        continue
                    amt_txt = (m2.group(1) or m2.group(2) or "").strip()
                    amt = _to_amount(amt_txt)
                    if amt is None:
                        continue
                    # остаток и описание — необязательные; выдернем «хвост»
                    tail = rest[m2.end():].strip()
                    rows.append({
                        "date": pd.to_datetime(dstr, dayfirst=True, errors="coerce"),
                        "amount": float(amt),
                        "type": "PRIJEM" if amt>0 else "VYDAJ",
                        "balance": None,
                        "counterparty": tail if tail else None
                    })
                continue  # следующая страница

            # === ВЕТКА Withdrawals/Deposits: найдём x-диапазоны колонок из заголовка ===
            # соберём мапу {colname -> (x_left, x_right)} из слов заголовка
            header_words = lines[header_idx]["words"]
            htxt = [w[2] for w in header_words]
            # найдём позиции ключевых слов
            def find_x(keywords):
                for (x0, x1, txt) in header_words:
                    t = txt.lower()
                    if any(k in t for k in keywords):
                        return (x0, x1)
                return None
            x_date = find_x(["date","datum","дата"])
            x_with = find_x(["withdraw", "withdrawals"])
            x_depo = find_x(["deposit","deposits","paid in"])
            x_bal  = find_x(["balance"])
            # если ничего не нашли — на эту страницу не применяем
            if x_date is None:
                continue
            # под колонки используем «интервалы» начиная от центра заголовочного слова
            def center(x): return 0.5*(x[0]+x[1]) if x else None
            cx_with, cx_depo, cx_bal = center(x_with), center(x_depo), center(x_bal)

            # пробегаем строки ниже заголовка
            for L in lines[header_idx+1:]:
                ws = L["words"]
                # первая «ячейка» — дата (два формата: dd Mon / Mon dd)
                txt_line = " ".join(w[2] for w in ws)
                # пропускаем «Closing balance», «Opening balance»…
                if re.search(r"closing balance|opening balance", txt_line, flags=re.I):
                    continue
                # выделим токены-числа по x-колонкам
                date_txt = None
                w_amt, d_amt, b_amt = None, None, None  # withdrawals, deposits, balance
                # дата — ищем MonthName Day или Day MonthName
                parts = [w for w in ws if re.match(r"(?i)^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|^\d{1,2}\.$", w[2])]
                # упрощённо: берём первые два слова строки как «дату»
                if len(ws) >= 2:
                    cand = f"{ws[0][2]} {ws[1][2]}"
                    # попробуем распарсить "Dec 30" (без года) — подставим ближайший разумный
                    m = re.match(r"(?i)^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})$", cand)
                    if m:
                        mon = _MON_MAP[m.group(1).lower()]
                        dd  = int(m.group(2))
                        date_txt = pd.Timestamp(year=pd.Timestamp.today().year, month=mon, day=dd)
                # распределим числовые токены по столбцам по x-координате
                for (x0,x1,txt) in ws:
                    if not re.search(r"[\d\.\,]", txt):
                        continue
                    val = _to_amount(txt)
                    if val is None:
                        continue
                    cx = 0.5*(x0+x1)
                    if cx_bal and cx > cx_bal - 5:  # правее баланса
                        b_amt = val; continue
                    if cx_depo and (cx_with is None or abs(cx - cx_depo) < abs(cx - cx_with)):
                        d_amt = val; continue
                    if cx_with:
                        w_amt = val; continue
                # соберём строку
                if date_txt and (w_amt is not None or d_amt is not None):
                    amt = (d_amt or 0.0) - (w_amt or 0.0)
                    rows.append({
                        "date": date_txt,
                        "amount": float(amt),
                        "type": "PRIJEM" if amt>0 else "VYDAJ",
                        "balance": float(b_amt) if b_amt is not None else None,
                        "counterparty": None
                    })
    # финализация
    if not rows:
        return pd.DataFrame(columns=["date","amount","type","balance","counterparty"])
    df = pd.DataFrame(rows)
    # чистка
    df = df.dropna(subset=["date","amount"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def pdf_to_transactions(path: str) -> pd.DataFrame:
    rows = []

    # --- 1) pdfplumber (линии -> текст) ---
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                for tbl in page.extract_tables() or []:
                    rows.extend(_collect_rows_from_matrix(tbl))
                if not rows:
                    tbls = page.extract_tables(table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "explicit_vertical_lines": [],
                        "explicit_horizontal_lines": [],
                        "snap_tolerance": 3,
                        "snap_x_tolerance": 3,
                        "snap_y_tolerance": 3,
                        "join_tolerance": 3,
                        "edge_min_length": 3,
                        "min_words_vertical": 2,
                        "min_words_horizontal": 1,
                    })
                    for tbl in tbls or []:
                        rows.extend(_collect_rows_from_matrix(tbl))
    except Exception:
        pass

    # --- 2) Camelot (lattice -> stream) ---
    if not rows:
        try:
            import camelot
            all_tbls = []
            for flavor in ("lattice","stream"):
                try:
                    t = camelot.read_pdf(path, pages="all", flavor=flavor)
                    if t and t.n > 0:
                        all_tbls.extend([ti.df for ti in t])
                except Exception:
                    continue
            for df in all_tbls:
                df = df.dropna(how="all")
                if df.empty: continue
                hdr = df.iloc[0].astype(str).tolist()
                body = df.iloc[1:].to_dict(orient="records")
                matrix = [hdr] + [[str(v) for v in rec.values()] for rec in body]
                rows.extend(_collect_rows_from_matrix(matrix))
        except Exception:
            pass

    # --- 3) tabula-py (Java) ---
    if not rows:
        try:
            import tabula
            # сначала "lattice" (есть линии), потом "stream" (по расстояниям между словами)
            for opt in ({"lattice": True}, {"stream": True}):
                dfs = tabula.read_pdf(
                    path, pages="all", multiple_tables=True,
                    pandas_options={"dtype": str}, **opt
                )
                for d in dfs or []:
                    if d is None or d.empty:
                        continue
                    d = d.dropna(how="all")
                    hdr = d.columns.astype(str).tolist()
                    body = d.fillna("").astype(str).values.tolist()
                    matrix = [hdr] + body
                    rows.extend(_collect_rows_from_matrix(matrix))
                if rows:
                    break
        except Exception:
            pass

    # --- 4) ЧИСТО-ТЕКСТОВЫЙ ФОЛБЭК (extract_words + координаты) ---
    if not rows:
        try:
            df_text = _parse_text_fallback(path)
            if not df_text.empty:
                return df_text
        except Exception:
            pass

    df = pd.DataFrame(rows, columns=["date", "amount", "type", "balance", "counterparty"])
    if not df.empty:
        df = df.dropna(subset=["date", "amount"]).reset_index(drop=True)
    return df
