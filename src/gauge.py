import math
import matplotlib.pyplot as plt

def save_gauge(score, min_s=1, max_s=1000, path="artifacts/credit_gauge.png", title="Credit Score"):
    # зоны
    bands = [
        (min_s, 500, "red"),
        (501, 700, "orange"),
        (701, 850, "yellow"),
        (851, max_s, "green"),
    ]
    fig, ax = plt.subplots(figsize=(6,3))
    ax.axis("off")

    # дуга
    theta0, theta1 = -math.pi * 0.75, math.pi * 0.75
    for lo, hi, color in bands:
        a = (lo - min_s) / (max_s - min_s)
        b = (hi - min_s) / (max_s - min_s)
        t0 = theta0 + (theta1 - theta0) * a
        t1 = theta0 + (theta1 - theta0) * b
        ts = [t0 + i*(t1-t0)/100 for i in range(101)]
        xs = [0.0 + 1.0*math.cos(t) for t in ts]
        ys = [0.0 + 1.0*math.sin(t) for t in ts]
        ax.plot(xs, ys, linewidth=12, solid_capstyle="butt", color=color)

    # стрелка
    a = (score - min_s) / (max_s - min_s)
    t = theta0 + (theta1 - theta0) * a
    ax.plot([0, 0.9*math.cos(t)], [0, 0.9*math.sin(t)], linewidth=3)
    ax.scatter([0],[0], s=30, zorder=5)

    # подписи
    ax.text(0, -0.2, f"{int(score)}", ha="center", va="center", fontsize=16)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
