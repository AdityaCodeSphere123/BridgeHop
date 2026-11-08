"""Try many small answer rewrites on every saved server run; list the ones that never lose (zero LLM calls)."""
import json, os, re, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.evaluate import answer_scores
gold = {g["id"]: g for g in map(json.loads, open("data/dev.jsonl"))}
runs = [d for d in glob.glob("runs/*/predictions.jsonl") if "kreplay" not in d and "/E/" not in d and "/b1/" not in d]
data = {r: {p["id"]: p for p in map(json.loads, open(r))} for r in runs}
data = {r: v for r, v in data.items() if len(v) == 300}
F = lambda a, g: answer_scores(a, [g["answer"]] + g["answer_aliases"])[1]
Q = lambda q, pat: re.match(pat, q.strip(), re.I)
R = {
 "language: drop ' language'":      lambda q, a: re.sub(r"\s+language$", "", a, flags=re.I) if Q(q, r".*\blanguage\b") else a,
 "drop 'the city of'":             lambda q, a: re.sub(r"^(the\s+)?city of\s+", "", a, flags=re.I),
 "drop 'Kingdom of'/'Republic of'": lambda q, a: re.sub(r"^(the\s+)?(kingdom|republic|state|province) of\s+", "", a, flags=re.I),
 "who: drop title Dr/Sir/King":     lambda q, a: re.sub(r"^(dr\.?|sir|king|queen|president|saint|st\.)\s+", "", a, flags=re.I) if Q(q, r"who") else a,
 "when: full date -> year if 'what year'": lambda q, a: (re.search(r"\b(1\d{3}|20\d{2})\b", a).group(1) if Q(q, r"(in )?what year|which year") and re.search(r"\b(1\d{3}|20\d{2})\b", a) else a),
 "drop trailing ' Records'":        lambda q, a: re.sub(r"\s+records$", "", a, flags=re.I),
 "drop trailing ' Inc.'/' Ltd'":    lambda q, a: re.sub(r",?\s+(inc\.?|ltd\.?|llc|co\.)$", "", a, flags=re.I),
 "drop trailing ' Band'/' band'":   lambda q, a: re.sub(r"\s+band$", "", a, flags=re.I),
 "drop leading 'about'/'around'":   lambda q, a: re.sub(r"^(about|around|roughly|approximately|circa|c\.)\s+", "", a, flags=re.I),
 "drop trailing 'years old'":       lambda q, a: re.sub(r"\s+years?\s+old$", "", a, flags=re.I),
 "drop trailing ' years'":          lambda q, a: re.sub(r"(?<=\d)\s+years?$", "", a, flags=re.I) if Q(q, r"how old") else a,
 "drop 'the' + keep (control)":     lambda q, a: a,
 "drop leading 'in '/'on '":        lambda q, a: re.sub(r"^(in|on|at|by)\s+(?=[A-Z0-9])", "", a),
 "drop trailing ' (film)' etc":     lambda q, a: re.sub(r"\s*\((film|song|album|band|novel|TV series|play)\)$", "", a, flags=re.I),
 "drop 'U.S. state of'":            lambda q, a: re.sub(r"^(the\s+)?(u\.?s\.?|us) state of\s+", "", a, flags=re.I),
 "drop trailing ' Province'":       lambda q, a: re.sub(r"\s+province$", "", a, flags=re.I),
 "drop trailing ' City' (not 'X City' gold)": lambda q, a: a,
 "drop leading 'St.' -> 'Saint'":   lambda q, a: re.sub(r"^St\.\s", "Saint ", a),
 "drop ' and ...' tail for who":    lambda q, a: a.split(" and ")[0] if Q(q, r"who\b") and " and " in a and "," not in a else a,
 "how many: number word->digit":    lambda q, a: {"one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}.get(a.lower(), a) if Q(q, r"how many") else a,
 "how many: digit->number word":    lambda q, a: {"1":"one","2":"two","3":"three","4":"four","5":"five","6":"six","7":"seven","8":"eight","9":"nine","10":"ten","11":"eleven","12":"twelve","13":"thirteen","14":"fourteen","15":"fifteen"}.get(a, a) if Q(q, r"how many") else a,
}
for name, f in R.items():
    wins = losses = 0; tot = 0.0; ch = set()
    for r, P in data.items():
        d = 0.0
        for i, p in P.items():
            a2 = f(gold[i]["question"], p["answer"])
            if a2 != p["answer"]:
                dd = F(a2, gold[i]) - F(p["answer"], gold[i]); d += dd
                if abs(dd) > 1e-9: ch.add((p["answer"][:30], a2[:30], gold[i]["answer"][:25], round(dd, 2)))
        tot += d; wins += d > 1e-9; losses += d < -1e-9
    print(f"{name:42s} better {wins:2d} / worse {losses:2d} of {len(data)}  sum dF1 {tot:+.2f}  examples {sorted(ch)[:3]}")
