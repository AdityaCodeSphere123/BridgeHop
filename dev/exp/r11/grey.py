"""Zero-token check of grey-area answer-format rules: build agent variants with one rule removed,
score each from runs/llm_cache.jsonl (misses never call the server)."""
import os, sys, importlib
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "dev/exp/r10")); sys.path.insert(0, ROOT); os.chdir(ROOT)
import ablate
from a4.data import SearchEngine, read_jsonl

SRC = open(sys.argv[1] if len(sys.argv) > 1 else "agent.py").read()
CUTS = {
 "century": ['    if re.search(r"\\bcentur(y|ies)\\b", question, re.I):', '    answer_text = final_form('],
}
def without(src, a, b):
    i = src.index(a); j = src.index(b, i); return src[:i] + src[j:]
def rep(src, a, b):
    assert a in src, a; return src.replace(a, b)
V = {}
V["century"] = without(SRC, *CUTS["century"])
V["how_long"] = rep(SRC, '    if long_form and re.match(r"\\s*how long\\b", question, re.I):', '    if False:')
V["month1"] = rep(SRC, '    ans = re.sub(r"^1\\s+(year|month|week|day|hour|decade|century)$", r"\\1", ans, flags=re.I)', '')
V["weeklong"] = rep(SRC, '    ans = re.sub(r"^(?:a |an |one )?(\\w+)\\s*-\\s*long$", r"\\1", ans, flags=re.I)', '')
V["for_cut"] = rep(SRC, '    if sep and len(head.split()) >= 2 and not re.search(r"\\d", ans):\n        ans = head\n    group', '    group')
V["people"] = rep(SRC, '    if group and norm(group.group(1)):', '    if False:')
V["county_append"] = rep(SRC, '        final_text = f"{final} County"', '        final_text = final')
V["comma_cut"] = rep(SRC, '    if sep and re.match(r"[A-Za-z]", tail) and "." not in tail and not re.search(r"\\b(and|or)\\b|&", ans):\n        ans = head', '')
V["hedge"] = rep(SRC, 'return re.sub(r"^(over|about|approximately|around|nearly|almost|more than|less than)\\s+(?=\\d)", "", ans,\n                  flags=re.I)', 'return ans')
V["acronym"] = rep(SRC, '    if not re.fullmatch(r"[A-Z]{2,6}", ans):', '    if True:')
V["drop_unit"] = rep(SRC, '    if not re.match(r"how many\\b", question.strip(), re.I):\n        return ans\n    m =', '    return ans\n    m =')
gold = list(read_jsonl("data/dev.jsonl")); engine = SearchEngine.load("data")
os.makedirs("dev/exp/r11/v", exist_ok=True)
def go(name, src):
    p = f"dev/exp/r11/v/{name}.py"; open(p, "w").write(src); os.environ["AGENT"] = p
    return ablate.run("full", gold, engine)
base, bm = go("base", SRC)
print(f"base S={sum(v[1] for v in base.values())/300:.4f} misses={len(bm)}")
names = sys.argv[2:] or list(V)
for n in names:
    res, mq = go(n, V[n])
    ch = [(i, base[i][0], res[i][0], round(res[i][1]-base[i][1],3)) for i in res if abs(res[i][1]-base[i][1]) > 1e-9]
    print(f"\n-{n}: S={sum(v[1] for v in res.values())/300:.4f} dS={sum(v[1] for v in res.values())/300 - sum(v[1] for v in base.values())/300:+.4f} misses={len(mq)} changed={len(ch)}")
    for c in ch[:8]: print("    ", c)
