# Development workflow (not part of the submission)

`make_submission.py` zips only `agent.py`, `helpers/` and `report.pdf`, so nothing in `dev/` or `report/`
is ever submitted.

| File | Purpose |
|---|---|
| `dev/b1_agent.py` | The untouched starter agent (baseline B1), for the report's comparison table |
| `dev/analyze.py` | Retrieval recall, bridge accuracy, automatic error categories (i)–(v) → `analysis.md` |
| `dev/compare.py` | Report table (Markdown + LaTeX rows) and per-question wins/losses between two runs |
| `report/report.tex` | Report template with the design already described; fill in the TODOs |

## Once the data bundle and API key arrive

```bash
# 0. setup: unpack the bundle into data/, then create config.json (see config.example.json)
python run_dev.py --mock-llm --limit 5                    # plumbing only, no server calls
python run_dev.py --limit 5                               # real LLM: confirms the key works

# 1. the two runs for the report table
python run_dev.py --agent dev/b1_agent.py --out runs/b1
python run_dev.py --out runs/v1
python dev/compare.py runs/b1 runs/v1 --names B1 Ours

# 2. where does it fail?
python dev/analyze.py runs/v1
python run_dev.py --ids <question_id>                     # rerun one question, then read traces.jsonl

# 3. after each change to agent.py: rerun, compare against the previous version
python run_dev.py --out runs/v2 && python dev/compare.py runs/v1 runs/v2

# 4. leaderboard upload (max 3 per day) and final Moodle zip
python make_submission.py <entry_number> --no-report
python make_submission.py <entry_number>                  # needs report.pdf in the repo root
```

## What to tune first (constants and prompts at the top of `agent.py`)

1. **Answer format.** In `analyze.py`, category (iv) and partial-F1 answers show whether answers are too
   long (e.g. "June 5, 1962" vs "1962", or "Steve Hillage's wife"). Adjust `READ_PROMPT` / `clean_answer`.
2. **Bridge accuracy.** If the bridge is often wrong while the hop-1 paragraph *was* retrieved, the fix is
   in the read prompt. If the paragraph was not retrieved, try `K_SUBQ` / `K_COMPANION` (e.g. 8 / 5).
3. **Plan quality.** Check that the `plan:` lines in the traces produce sensible Q1 / Q2 pairs. If many
   fail to parse, add an in-context example shaped like the failing questions.
4. **Prompt size.** `MAX_PROMPT_PASSAGES` (10) versus fewer, cleaner passages: small models can get
   distracted by passages that are irrelevant.
5. **Time.** `analyze.py` prints the maximum seconds per question. Keep it well under 60, because the
   shared server will be slower near the deadline.
