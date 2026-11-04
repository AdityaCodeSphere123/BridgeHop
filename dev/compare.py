#!/usr/bin/env python3
"""Compare runs of run_dev.py and print the report's required table (development tool - NOT submitted).

    python dev/compare.py runs/<b1_run> runs/<agent_run> [more runs ...] --names B1 Ours

Prints the table (EM, answer F1, evidence F1, mean search / LLM calls, S) as Markdown and as LaTeX rows,
then, for the last two runs, how many questions each one wins and a few examples of each, which helps
to tell whether a change really helped or just moved errors around.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a4.data import read_jsonl  # noqa: E402


def load(run_dir):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)
    per_q = {r["id"]: r for r in read_jsonl(os.path.join(run_dir, "review.jsonl"))}
    return metrics, per_q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--names", nargs="*", default=[])
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()
    names = args.names + [os.path.basename(r.rstrip("/")) for r in args.runs[len(args.names):]]
    loaded = [load(r) for r in args.runs]

    cols = [("answer_em", "EM"), ("answer_f1", "Answer F1"), ("evidence_f1", "Evidence F1"),
            ("mean_search_calls", "Searches"), ("mean_llm_calls", "LLM calls"), ("S", "S")]
    print("| System | " + " | ".join(c for _, c in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for name, (m, _) in zip(names, loaded):
        print(f"| {name} | " + " | ".join(f"{m[k]:.4f}" if k not in ("mean_search_calls", "mean_llm_calls")
                                          else f"{m[k]:.2f}" for k, _ in cols) + " |")
    print("\nLaTeX rows:")
    for name, (m, _) in zip(names, loaded):
        print(f"{name} & " + " & ".join(f"{m[k]:.3f}" if k not in ("mean_search_calls", "mean_llm_calls")
                                       else f"{m[k]:.2f}" for k, _ in cols) + r" \\")

    if len(loaded) >= 2:
        (_, a), (_, b) = loaded[-2], loaded[-1]
        common = sorted(set(a) & set(b))
        score = lambda r: 0.8 * r["f1"] + 0.2 * r["ev_f1"]  # noqa: E731
        wins = [q for q in common if score(b[q]) > score(a[q]) + 1e-9]
        losses = [q for q in common if score(b[q]) < score(a[q]) - 1e-9]
        print(f"\n{names[-1]} vs {names[-2]} on {len(common)} questions: "
              f"{len(wins)} better, {len(losses)} worse, {len(common) - len(wins) - len(losses)} same")
        for title, ids in (("better", wins), ("worse", losses)):
            print(f"\nExamples where {names[-1]} is {title}:")
            for q in ids[: args.show]:
                print(f"  {q}: {b[q]['question']}\n     gold={b[q]['gold']!r}  "
                      f"{names[-2]}={a[q]['pred']!r}  {names[-1]}={b[q]['pred']!r}")


if __name__ == "__main__":
    main()
