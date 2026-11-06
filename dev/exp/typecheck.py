"""Accepted hop-2 answers in a recorded run (no LLM): how good are the ones that fail a type check or merely
repeat words of the question?"""
import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.evaluate import answer_scores, normalize_answer as N
G = {json.loads(l)["id"]: json.loads(l) for l in open("data/dev.jsonl")}
P = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1] + "/predictions.jsonl")}
T = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1] + "/traces.jsonl")}
MONTHS = r"january|february|march|april|may|june|july|august|september|october|november|december"
def expected(q):
    ql = q.lower()
    if re.search(r"\bcounty\b", ql) and re.search(r"\b(what|which|in what|in which)\b[^?]*\bcounty\b|\bcounty\b[^?]*\?$", ql) and "county" not in ql.split("county")[0][-0:]:
        return "county"
    if re.match(r"\s*(when|what year|which year|in what year|in which year|what date)\b", ql) or re.search(r"\b(what|which) year\b", ql):
        return "time"
    if re.match(r"\s*how (many|much|long|old|tall|big|large|high)\b", ql) or re.search(r"\bpopulation\b", ql):
        return "number"
    return None
def ok(t, a):
    al = a.lower()
    if t == "county": return bool(re.search(r"\b(county|parish|borough|shire)\b", al))
    if t == "time": return bool(re.search(r"\d", al) or re.search(MONTHS, al) or re.search(r"\b(century|centuries|decade|era|age|war|period|dynasty)\b", al))
    if t == "number": return bool(re.search(r"\d", al) or re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|hundred|thousand|million|billion|dozen|once|twice)\b", al))
    return True
from collections import defaultdict
st = defaultdict(lambda: [0, 0.0]); ex = defaultdict(list)
for qid, t in T.items():
    g = G[qid]; golds = [g["answer"]] + g["answer_aliases"]; q = g["question"]
    f1 = answer_scores(P[qid]["answer"], golds)[1]
    typ = expected(q)
    a = P[qid]["answer"]
    key = f"type={typ}: " + ("OK" if ok(typ, a) else "MISMATCH") if typ else "type=None"
    st[key][0] += 1; st[key][1] += f1
    if typ and not ok(typ, a): ex[key].append((q[:80], a, g["answer"]))
    inq = bool(N(a)) and f" {N(a)} " in f" {N(q)} "
    k2 = "answer inside question" if inq else "answer not in question"
    st[k2][0] += 1; st[k2][1] += f1
    if inq: ex[k2].append((q[:80], a, g["answer"]))
for k, (n, f) in sorted(st.items()): print(f"{k:32s} n={n:3d} meanF1={f/n:.3f}")
for k, v in ex.items():
    print("\n##", k)
    for e in v[:14]: print("  ", e)
