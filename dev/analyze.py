#!/usr/bin/env python3
"""Error analysis for one run of run_dev.py (development tool - NOT part of the submission).

    python dev/analyze.py runs/<timestamp>                 # uses data/dev.jsonl and data/corpus.jsonl
    python dev/analyze.py runs/<timestamp> --show 15       # print more example errors

What it reports
  * retrieval diagnostics: was each gold paragraph retrieved by *any* search? both? which search found it?
  * bridge accuracy: does the agent's hop-1 answer match the gold bridge (from dev.jsonl's decomposition)?
  * an automatic label for every wrong question, using the handout's report categories:
      (i)   bridge entity not found          (iii) evidence retrieved but answered wrongly
      (ii)  stopped too early                (iv)  answer correct but marked wrong (partial / alias match)
      (v)   other (crash, timeout, hop-2 paragraph never retrieved despite the right bridge, ...)
    The labels are heuristics: check the ones you put in the report by hand (traces.jsonl has everything).

Writes <run>/analysis.md and <run>/errors.jsonl (one labelled line per wrong question).
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a4.data import read_jsonl  # noqa: E402
from a4.evaluate import answer_scores, normalize_answer  # noqa: E402

CATEGORIES = {
    "i": "bridge entity not found",
    "ii": "stopped too early",
    "iii": "evidence retrieved but answered wrongly",
    "iv": "answer correct but marked wrong (format / alias)",
    "v": "other",
}
HOP1_MARKER = "This is one step towards answering"   # text of the agent's hop-1 read prompt


def load_corpus_texts(path, wanted):
    """doc_id -> 'title text' for the doc_ids in ``wanted`` only (keeps memory small)."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["doc_id"] in wanted:
                out[r["doc_id"]] = r["title"] + " " + r["text"]
    return out


def gold_bridge(g):
    """Answer of the first gold sub-question, if dev.jsonl provides a decomposition."""
    dec = g.get("decomposition") or g.get("question_decomposition") or []
    if isinstance(dec, list) and dec and isinstance(dec[0], dict):
        return dec[0].get("answer") or ""
    return ""


def contains(text, phrase):
    phrase = normalize_answer(phrase or "")
    return bool(phrase) and f" {phrase} " in f" {normalize_answer(text)} "


def agent_bridge(trace):
    """The agent's last non-unknown hop-1 answer, parsed from its LLM calls."""
    best = ""
    for c in trace["calls"]:
        if c["type"] == "llm" and HOP1_MARKER in c.get("prompt", ""):
            m = re.search(r"Answer\s*:\s*(.+)", c.get("output") or "")
            if m and "unknown" not in m.group(1).lower():
                best = m.group(1).strip(" .")
    return best


def label(row):
    """Heuristic report category for a wrong answer (see module docstring)."""
    if row["status"] != "ok":
        return "v", f"status={row['status']}"
    pred, golds = normalize_answer(row["pred"]), [normalize_answer(a) for a in row["golds"]]
    if row["f1"] > 0 or any(pred and (pred in a or a in pred) for a in golds):
        return "iv", "partial overlap with gold - check whether it is really correct"
    if row["both_retrieved"]:
        return "iii", "both gold paragraphs were retrieved"
    if row["bridge_ok"] is False or (row["hop1_doc"] and not row["hop1_retrieved"]):
        return "i", f"agent bridge {row['agent_bridge']!r} vs gold {row['gold_bridge']!r}"
    if row["n_search"] < 5:
        return "ii", f"stopped after {row['n_search']} searches without the hop-2 paragraph"
    return "v", "right bridge, but the hop-2 paragraph was never retrieved within 5 searches"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="dev.jsonl")
    ap.add_argument("--show", type=int, default=8, help="example errors to print per category")
    args = ap.parse_args()

    gold = {g["id"]: g for g in read_jsonl(os.path.join(args.data_dir, args.split))}
    preds = {r["id"]: r for r in read_jsonl(os.path.join(args.run_dir, "predictions.jsonl"))}
    traces = {t["id"]: t for t in read_jsonl(os.path.join(args.run_dir, "traces.jsonl"))}
    wanted = {d for i in preds for d in gold[i]["supporting_doc_ids"]}
    texts = load_corpus_texts(os.path.join(args.data_dir, "corpus.jsonl"), wanted)

    rows = []
    for qid, r in preds.items():
        g, t = gold[qid], traces.get(qid, {"calls": []})
        golds = [g["answer"]] + list(g.get("answer_aliases", []))
        em, f1 = answer_scores(r["answer"], golds) if r["status"] == "ok" else (0.0, 0.0)
        searches = [c for c in t["calls"] if c["type"] == "search"]
        retrieved = {d for c in searches for d in c["doc_ids"]}
        gids = list(g["supporting_doc_ids"])
        # The hop-2 paragraph is the gold paragraph that states the final answer; the other one is hop 1.
        hop2 = next((d for d in gids if any(contains(texts.get(d, ""), a) for a in golds)), None)
        hop1 = next((d for d in gids if d != hop2), None) if hop2 else None
        gb, ab = gold_bridge(g), agent_bridge(t)
        rows.append({
            "id": qid, "question": g["question"], "status": r["status"], "pred": r["answer"],
            "gold": g["answer"], "golds": golds, "em": em, "f1": f1,
            "citations": r["citations"], "gold_doc_ids": gids,
            "cit_hits": len(set(r["citations"][:3]) & set(gids)),
            "retrieved_gold": [d in retrieved for d in gids],
            "both_retrieved": all(d in retrieved for d in gids),
            "hop1_doc": hop1, "hop1_retrieved": hop1 in retrieved if hop1 else None,
            "hop2_retrieved": hop2 in retrieved if hop2 else None,
            "gold_bridge": gb, "agent_bridge": ab,
            "bridge_ok": (answer_scores(ab, [gb])[1] >= 0.5) if gb else None,
            "n_search": r["n_search"], "n_llm": r["n_llm"], "elapsed": r.get("elapsed", 0.0),
            "queries": [c["query"] for c in searches],
        })

    n = len(rows)
    pct = lambda k: 100.0 * k / max(1, n)  # noqa: E731
    wrong = [x for x in rows if x["em"] < 1.0]
    for x in wrong:
        x["category"], x["why"] = label(x)
    cats = Counter(x["category"] for x in wrong)
    with_bridge = [x for x in rows if x["bridge_ok"] is not None]
    both = [x for x in rows if x["both_retrieved"]]
    elapsed = sorted(x["elapsed"] for x in rows)

    lines = [f"# Analysis of {args.run_dir}", "", f"questions: {n}", "",
             "## Retrieval", "",
             f"* any gold paragraph retrieved:  {pct(sum(any(x['retrieved_gold']) for x in rows)):.1f}%",
             f"* both gold paragraphs retrieved: {pct(len(both)):.1f}%",
             f"* hop-1 paragraph retrieved:      {pct(sum(bool(x['hop1_retrieved']) for x in rows)):.1f}%",
             f"* hop-2 paragraph retrieved:      {pct(sum(bool(x['hop2_retrieved']) for x in rows)):.1f}%",
             f"* citations: mean gold hits {sum(x['cit_hits'] for x in rows) / max(1, n):.2f} of 2",
             "", "## Reasoning", ""]
    if with_bridge:
        lines.append(f"* bridge (hop-1 answer) correct: "
                     f"{100.0 * sum(x['bridge_ok'] for x in with_bridge) / len(with_bridge):.1f}%")
    if both:
        lines.append(f"* answer F1 when both gold paragraphs were retrieved: "
                     f"{sum(x['f1'] for x in both) / len(both):.3f}  (n={len(both)})")
    lines += [f"* searches used: {dict(sorted(Counter(x['n_search'] for x in rows).items()))}",
              f"* LLM calls used: {dict(sorted(Counter(x['n_llm'] for x in rows).items()))}",
              f"* seconds per question: median {elapsed[n // 2] if n else 0:.1f}, "
              f"max {elapsed[-1] if n else 0:.1f} (the limit is 60)",
              "", f"## Errors (EM = 0): {len(wrong)} questions", ""]
    for c, desc in CATEGORIES.items():
        lines.append(f"* ({c}) {desc}: {cats.get(c, 0)}")
    for c in CATEGORIES:
        examples = [x for x in wrong if x["category"] == c][: args.show]
        if not examples:
            continue
        lines += ["", f"### ({c}) {CATEGORIES[c]}", ""]
        for x in examples:
            lines += [f"* `{x['id']}` {x['question']}",
                      f"  * gold {x['gold']!r} | pred {x['pred']!r} | F1 {x['f1']:.2f}",
                      f"  * bridge: agent {x['agent_bridge']!r} / gold {x['gold_bridge']!r} | {x['why']}",
                      f"  * queries: {x['queries']}"]
    report = "\n".join(lines) + "\n"
    print(report)
    with open(os.path.join(args.run_dir, "analysis.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(args.run_dir, "errors.jsonl"), "w", encoding="utf-8") as f:
        for x in wrong:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"wrote {args.run_dir}/analysis.md and errors.jsonl")


if __name__ == "__main__":
    main()
