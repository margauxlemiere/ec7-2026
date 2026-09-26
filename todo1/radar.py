import numpy as np
import matplotlib.pyplot as plt

# ---- 1. les dix criteres, dans l'ordre du tableau de grid.md ----
criteria = [
    "Latence",
    "Cout total",
    "Energie",
    "Flux / complet",
    "Deploiement /\nsouverainete",
    "Robustesse",
    "Explicabilite",
    "Confidentialite",
    "Maintenabilite",
    "Controle humain",
]

# ---- 2. les scores (-3 a +3) par option, meme ordre que criteria ----
scores = {
    "A - grammaire fermee":  [3, 3, 3, 1, 3, -2, 3, 3, 2, 3],
    "B - ASR local + regles": [2, 1, 1, 2, 2, 1, 2, 3, 1, 1],
    "C - API cloud":          [-2, -1, -1, -2, -3, 2, -2, -3, -2, -2],
}

# ---- 3. mise en forme radar ----
n = len(criteria)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
angles += angles[:1]  # fermer le cercle

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# axes gradues de -3 a +3
ax.set_ylim(-3, 3)
ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_yticklabels(["-3", "-2", "-1", "0", "+1", "+2", "+3"], fontsize=8)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(criteria, fontsize=9)

colors = {
    "A - grammaire fermee":  "#1b7837",
    "B - ASR local + regles": "#2166ac",
    "C - API cloud":          "#b2182b",
}

for label, values in scores.items():
    values_closed = values + values[:1]
    ax.plot(angles, values_closed, label=label, color=colors[label], linewidth=2)
    ax.fill(angles, values_closed, color=colors[label], alpha=0.08)

# ligne du zero, bien visible (0 = ni aide ni gene)
ax.plot(angles, [0] * (n + 1), color="grey", linewidth=1, linestyle="--", alpha=0.6)

ax.set_title("Radar - options A / B / C sur les dix criteres", fontsize=13, pad=30)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

plt.tight_layout()
plt.savefig("radar.png", dpi=200, bbox_inches="tight")
print("radar.png genere.")
