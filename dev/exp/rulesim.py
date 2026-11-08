"""Rescore every saved server run after rewriting its final answers with a candidate rule (zero LLM calls).

    python3 dev/exp/rulesim.py
Each rule is f(question, answer) -> answer. Prints, per rule, how many runs gain / lose and the total
answer-F1 change summed over runs, plus the questions it changes in the current run.
"""
import json, os, re, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.evaluate import answer_scores
gold = {g["id"]: g for g in map(json.loads, open("data/dev.jsonl"))}
runs = sorted(d for d in glob.glob("runs/*/predictions.jsonl") if "kreplay" not in d and "/E/" not in d and "/b1/" not in d)
data = {r: {p["id"]: p for p in map(json.loads, open(r))} for r in runs}
data = {r: v for r, v in data.items() if len(v) == 300}
F = lambda a, g: answer_scores(a, [g["answer"]] + g["answer_aliases"])[1]
def evaluate(name, rule, show="runs/FINAL_full6/predictions.jsonl"):
    wins = losses = 0; total = 0.0
    for r, P in data.items():
        d = sum(F(rule(gold[i]["question"], p["answer"]), gold[i]) - F(p["answer"], gold[i]) for i, p in P.items())
        total += d; wins += d > 1e-9; losses += d < -1e-9
    print(f"{name:34s} runs better {wins:2d} / worse {losses:2d} of {len(data)}  sum dF1 {total:+.2f}  (dS per run {0.8*total/len(data)/300:+.4f})")
    for i, p in data[show].items():
        a = rule(gold[i]["question"], p["answer"])
        if a != p["answer"]:
            print(f"      {F(p['answer'], gold[i]):.2f}->{F(a, gold[i]):.2f}  {p['answer'][:40]!r} -> {a[:40]!r}  gold={gold[i]['answer'][:30]!r}")
RULES = {}
def rule(f): RULES[f.__name__] = f; return f
@rule
def letter_hyphen_join(q, a):  # "Portuguese - based" -> "Portuguese-based" (keep spaced hyphens between numbers)
    return re.sub(r"(?<=[A-Za-z]) - (?=[A-Za-z])", "-", a)
@rule
def drop_paren(q, a):
    return re.sub(r"\s*\((?![^)]*\d)[^)]*\)\s*$", "", a).strip() or a
@rule
def drop_type_noun(q, a):  # "What forest ..." + "Kabaty Forest" -> "Kabaty" (not county)
    m = re.match(r"\s*(?:what|which)\s+(\w+)", q, re.I)
    if m and m.group(1).lower() not in {"county", "parish", "borough"} and len(a.split()) >= 2 and a.split()[-1].lower() == m.group(1).lower():
        return " ".join(a.split()[:-1])
    return a
@rule
def drop_trailing_period(q, a):
    return a.rstrip(".") if a.endswith(".") and not re.search(r"\b[A-Z]\.$", a) else a
@rule
def and_list_first(q, a):  # "A, B, and C" for a "who" question -> keep it (control)
    return a
for n, f in RULES.items(): evaluate(n, f)
