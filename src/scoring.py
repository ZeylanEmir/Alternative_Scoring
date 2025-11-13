import math

def pd_to_score(pd: float, min_score=1, max_score=1000, base_points=600.0, base_odds=20.0, pdo=50.0) -> float:
    """
    Преобразование вероятности дефолта (pd) в скор по лог-оддс (PDO scheme).
    base_points — баллы при odds = base_odds (например, 20:1)
    pdo — сколько баллов за удвоение odds.
    """
    pd = max(1e-9, min(1 - 1e-9, float(pd)))
    odds = (1 - pd) / pd
    factor = pdo / math.log(2.0)
    offset = base_points - factor * math.log(base_odds)
    raw = offset + factor * math.log(odds)
    return max(min_score, min(max_score, raw))
