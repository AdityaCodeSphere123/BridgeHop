"""Zero-token ablation: run agent.py in-process on all dev questions with the LLM served only from
runs/llm_cache.jsonl (a cache miss raises LLMError, so the server is never called). Each variant switches
one feature off; prints S, misses and per-question wins/losses against the full agent.

    python3 dev/exp/r10/ablate.py [variant ...]
"""
import hashlib, importlib.util, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.budget import LLMError
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import answer_scores, evidence_f1
from a4.tools import Tools

cache = {}
for line in open("runs/llm_cache.jsonl"):
    try:
        r = json.loads(line); cache[r["k"]] = r["v"]
    except Exception:
        pass
key = lambda *parts: hashlib.sha1(json.dumps(parts).encode()).hexdigest()

class CacheLLM:
    misses = 0
    def count_tokens(self, prompt):
        v = cache.get(key("t", prompt))
        if v is None:
            CacheLLM.misses += 1; raise LLMError("miss")
        return v
    def complete(self, prompt, max_tokens):
        v = cache.get(key("c", prompt, max_tokens))
        if v is None:
            CacheLLM.misses += 1; raise LLMError("miss")
        return tuple(v)

def load_agent():
    spec = importlib.util.spec_from_file_location("agent_probe", os.environ.get("AGENT", os.path.join(ROOT, "agent.py")))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def variant(name):
    A = load_agent()
    if name == "no_weak_recheck":
        A.recheck_weak_bridge = lambda q, n, t, c, s, a, ci: (a, ci)
    elif name == "no_place_type":
        A.wrong_place_type = lambda q, q1: False
    elif name == "no_copied_example":
        A.copied_example = lambda s, q: False
    elif name == "no_memory_check":
        A.memory_check = lambda q, n, t, c, asked, jq, a, ci: (a, ci)
    elif name == "no_when_q2d":
        orig = A.hypothetical_passage
        def hp(subq, tools, _o=orig):  # only the hop-2 "when" retry calls it from next_query at hop 2
            import inspect
            fr = inspect.stack()[1]
            if fr.function == "next_query" and fr.frame.f_locals["notes"].hop == 1:
                return ""
            return _o(subq, tools)
        A.hypothetical_passage = hp
    elif name == "no_hop1_q2d":
        orig = A.hypothetical_passage
        def hp(subq, tools, _o=orig):
            import inspect
            fr = inspect.stack()[1]
            if fr.function == "next_query" and fr.frame.f_locals["notes"].hop == 0:
                return ""
            return _o(subq, tools)
        A.hypothetical_passage = hp
    elif name == "no_for_people":
        src_ff = A.final_form
        def ff(ans, _A=A):
            import re
            ans = ans.replace("*", "").strip()
            ans = re.sub(r"^(?:a |an |one )?(\w+)\s*-\s*long$", r"\1", ans, flags=re.I)
            ans = re.sub(r"^1\s+(year|month|week|day|hour|decade|century)$", r"\1", ans, flags=re.I)
            head, sep, tail = ans.partition(", ")
            if sep and re.match(r"[A-Za-z]", tail) and "." not in tail and not re.search(r"\b(and|or)\b|&", ans):
                ans = head
            ans = re.sub(r"\s*[-–—]+\s*", " - ", ans)
            return re.sub(r"^(over|about|approximately|around|nearly|almost|more than|less than)\s+(?=\d)", "", ans, flags=re.I)
        A.final_form = ff
    elif name == "no_comma_cut":
        def ff(ans, _A=A):
            import re
            ans = ans.replace("*", "").strip()
            head, sep, _ = ans.partition(" for ")
            if sep and len(head.split()) >= 2 and not re.search(r"\d", ans):
                ans = head
            group = re.match(r"(.+?)\s+people$", ans, re.I)
            if group and _A.norm(group.group(1)):
                ans = group.group(1)
            ans = re.sub(r"^(?:a |an |one )?(\w+)\s*-\s*long$", r"\1", ans, flags=re.I)
            ans = re.sub(r"^1\s+(year|month|week|day|hour|decade|century)$", r"\1", ans, flags=re.I)
            ans = re.sub(r"\s*[-–—]+\s*", " - ", ans)
            return re.sub(r"^(over|about|approximately|around|nearly|almost|more than|less than)\s+(?=\d)", "", ans, flags=re.I)
        A.final_form = ff
    elif name == "no_drop_unit":
        A.drop_unit = lambda q, a: a
    elif name == "no_acronym":
        A.expand_acronym = lambda a, n: a
    elif name != "full":
        raise SystemExit("unknown variant " + name)
    return A

def run(name, gold, engine):
    A = variant(name); CacheLLM.misses = 0
    out, missq = {}, set()
    for g in gold:
        before = CacheLLM.misses
        t = Tools(engine, CacheLLM())
        r = A.run(g["question"], t)
        if CacheLLM.misses > before:
            missq.add(g["id"])
        golds = [g["answer"]] + list(g.get("answer_aliases", []))
        em, f1 = answer_scores(r["answer"], golds)
        ev = evidence_f1(r["citations"], g["supporting_doc_ids"])
        out[g["id"]] = (r["answer"], 0.8 * f1 + 0.2 * ev, f1)
    return out, missq

if __name__ == "__main__":
    gold = list(read_jsonl("data/dev.jsonl")); engine = SearchEngine.load("data")
    base, bm = run("full", gold, engine)
    print(f"full: S={sum(v[1] for v in base.values())/len(gold):.4f} misses={len(bm)}")
    for name in sys.argv[1:]:
        res, mq = run(name, gold, engine)
        clean = [i for i in res if i not in mq]
        wins = [(i, base[i][0], res[i][0]) for i in clean if res[i][1] > base[i][1] + 1e-9]
        loss = [(i, base[i][0], res[i][0]) for i in clean if res[i][1] < base[i][1] - 1e-9]
        d = sum(res[i][1] - base[i][1] for i in clean) / len(gold)
        print(f"\n{name}: dS(clean)={d:+.4f}  misses={len(mq)}  removing it wins {len(wins)} / loses {len(loss)}")
        for w in wins: print("   + ", w)
        for l in loss: print("   - ", l)
        if mq: print("   miss ids:", sorted(mq)[:12])
