#!/usr/bin/env python3
"""Draw the figures used in README.md from the measured dev-set results.

    python3 dev/make_figures.py          # writes docs/figures/*.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
os.makedirs(OUT, exist_ok=True)
BLUE, GREEN, GREY, ORANGE = "#2B6CB0", "#2F855A", "#A0AEC0", "#DD6B20"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=170)
    plt.close(fig)


# 1. Headline comparison against the two baselines (300 dev questions)
systems = ["Single-shot\n(1 search, 1 LLM call)", "Simple iterative\nbaseline", "This agent"]
metrics = {"Exact match": [0.1400, 0.2033, 0.4967], "Answer F1": [0.2110, 0.2596, 0.5965],
           "Evidence F1": [0.3717, 0.4767, 0.6650], "Score S": [0.2431, 0.3030, 0.6102]}
fig, ax = plt.subplots(figsize=(8, 3.6))
w = 0.2
for i, (name, vals) in enumerate(metrics.items()):
    xs = [j + (i - 1.5) * w for j in range(len(systems))]
    bars = ax.bar(xs, vals, w, label=name, color=[GREY, ORANGE, BLUE, GREEN][i])
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
ax.set_xticks(range(len(systems)), systems)
ax.set_ylim(0, 0.75)
ax.set_ylabel("dev score")
ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left")
ax.set_title("Two-hop QA on 300 MuSiQue dev questions")
save(fig, "comparison.png")

# 2. How the score grew as components were added
steps = ["first iterative agent", "+ fallbacks / citations / RRF", "+ plan check & retry prompt",
         "+ topic article first", "+ second read & judge", "+ original question in judge", "+ memory check",
         "+ Query2doc retries", "final: + recheck & plan checks"]
scores = [0.5086, 0.5213, 0.5402, 0.5521, 0.5667, 0.5757, 0.5896, 0.5992, 0.6102]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.barh(steps, scores, color=[BLUE] * (len(steps) - 1) + [GREEN])
for y, v in enumerate(scores):
    ax.text(v + 0.002, y, f"{v:.4f}", va="center", fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0.49, 0.64)
ax.set_xlabel("dev score S")
ax.set_title("Score as components were added")
save(fig, "progress.png")

# 3. Why the topic article goes first: the 4B reader depends on passage order
fig, ax = plt.subplots(figsize=(5, 3.2))
pos = ["1st", "2nd", "3rd or lower", "not shown"]
acc = [83, 71, 67, 28]
ax.bar(pos, acc, color=BLUE)
for x, v, n in zip(range(4), acc, [214, 24, 24, 32]):
    ax.text(x, v + 1.5, f"{v}%\n(n={n})", ha="center", fontsize=8)
ax.set_ylim(0, 100)
ax.set_ylabel("bridge correct (%)")
ax.set_title("First-hop accuracy by position of the right passage")
save(fig, "bridge_position.png")

# 4. Where the errors are: single-shot vs this agent
cats = ["(i) bridge not found", "(ii) stopped too early", "(iii) misread evidence", "(iv) right answer, other form",
        "(v) other"]
b1, final = [191, 0, 15, 52, 0], [35, 15, 42, 51, 8]
fig, ax = plt.subplots(figsize=(7, 3.2))
ys = range(len(cats))
ax.barh([y - 0.2 for y in ys], b1, 0.4, color=GREY, label="single-shot (258 errors)")
ax.barh([y + 0.2 for y in ys], final, 0.4, color=BLUE, label="this agent (151 errors)")
for y, a, b in zip(ys, b1, final):
    ax.text(a + 2, y - 0.2, str(a), va="center", fontsize=8)
    ax.text(b + 2, y + 0.2, str(b), va="center", fontsize=8)
ax.set_yticks(list(ys), cats)
ax.invert_yaxis()
ax.set_xlabel("questions with exact match 0")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Error types on the dev set")
save(fig, "errors.png")

# 5. Robustness: keep going when the LLM server fails a call
fig, ax = plt.subplots(figsize=(5, 3.2))
rates = [0, 5, 15]
ax.plot(rates, [0.6102, 0.6014, 0.5633], "o-", color=GREEN, label="continues after a failed call")
ax.plot(rates, [0.6102, 0.5391, 0.4260], "s-", color=ORANGE, label="stops at the first failure")
ax.set_xticks(rates)
ax.set_xlabel("LLM calls that fail (%)")
ax.set_ylabel("dev score S")
ax.legend(frameon=False, fontsize=8)
ax.set_title("Score under injected server failures")
save(fig, "robustness.png")

print("figures written to", OUT)
