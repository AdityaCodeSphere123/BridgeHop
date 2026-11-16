import os, sys
ROOT="/Users/aditya/Desktop/classes/COL764-Assignment4"
sys.path.insert(0, ROOT+"/dev/exp/r10"); sys.path.insert(0, ROOT); os.chdir(ROOT)
import ablate
from a4.data import SearchEngine, read_jsonl
gold = list(read_jsonl("data/dev.jsonl")); engine = SearchEngine.load("data")
def go(p):
    os.environ["AGENT"] = p; return ablate.run("full", gold, engine)
base, bm = go("dev/exp/r12/agent_298.py")
print(f"298 replay S={sum(v[1] for v in base.values())/300:.4f} misses={len(bm)}")
S = open("dev/exp/r12/agent_298.py").read()
def rep(a, b):
    assert a in S, a[:60]; return S.replace(a, b)
V = {
 "county_append": rep('        final_text = f"{final} County"', '        final_text = final'),
 "county_type": rep('        return bool(re.search(r"\\b(county|parish|borough|shire)\\b", a))', '        return True'),
 "final_form": rep('    ans = ans.replace("*", "").strip()\n    # "Riverton', '    return ans.replace("*", "").strip()\n    # "Riverton'),
 "drop_unit": rep('def drop_unit(question: str, ans: str) -> str:\n', 'def drop_unit(question: str, ans: str) -> str:\n    return ans\n'),
 "acronym": rep('def expand_acronym(ans: str, notes: Notes) -> str:\n', 'def expand_acronym(ans: str, notes: Notes) -> str:\n    return ans\n'),
 "acr_strip": rep('    line = re.sub(r"\\s*\\([A-Z][A-Z.&]{1,9}\\)$", "", line)', '    pass'),
}
os.makedirs("dev/exp/r12/v", exist_ok=True)
for n, src in V.items():
    p = f"dev/exp/r12/v/{n}.py"; open(p, "w").write(src)
    res, mq = go(p)
    ch = [(i, base[i][0], res[i][0], round(res[i][1]-base[i][1], 2)) for i in res if abs(res[i][1]-base[i][1]) > 1e-9]
    print(f"\n-{n}: dS={sum(v[1] for v in res.values())/300 - sum(v[1] for v in base.values())/300:+.4f} misses={len(mq)} changed={len(ch)}")
    for c in ch[:6]: print("    ", c)
