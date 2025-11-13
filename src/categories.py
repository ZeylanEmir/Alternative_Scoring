
RISKY_CATEGORIES = {"GAMBLING", "BETTING", "HIGH_RISK_CONTENT"}

# MCC → категория (частичный пример)
MCC_TO_CATEGORY = {
    "7995": "GAMBLING",       # betting/lottery
    "7801": "BETTING",        # пример, уточни при необходимости
    "4829": "P2P_TRANSFER",   # money orders/wire/P2P общая корзина
    "6011": "CASH_WITHDRAWAL",
    # Utilities/Taxes и т.д. — заполни по мере надобности
}

# Ключевые слова → категория (для выписок без MCC)
# Смотри на типичные наименования в твоих Kaspi/Halyk CSV/PDF
KEYWORDS_TO_CATEGORY = {
    "mostbet": "GAMBLING",
    "1xbet": "BETTING",
    "olympbet": "BETTING",
    "melbet": "BETTING",
    "bk": "BETTING",
    "steam": "HIGH_RISK_CONTENT",
    "qiwi": "P2P_TRANSFER",
    "kaspi перевод": "P2P_TRANSFER",
    "p2p": "P2P_TRANSFER",
    "наличные": "CASH_WITHDRAWAL",
    "снятие наличных": "CASH_WITHDRAWAL",
    "kegoc": "UTILITIES",
    "kazakhtelecom": "UTILITIES",
    "tax": "TAXES",
}

def normalize_text(s: str) -> str:
    return (s or "").lower()

def map_to_category(counterparty: str, mcc: str | None = None) -> str | None:
    # приоритет MCC, затем ключевые слова
    if mcc and mcc in MCC_TO_CATEGORY:
        return MCC_TO_CATEGORY[mcc]
    t = normalize_text(counterparty)
    for kw, cat in KEYWORDS_TO_CATEGORY.items():
        if kw in t:
            return cat
    return None
