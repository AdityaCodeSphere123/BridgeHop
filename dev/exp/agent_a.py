"""COL 764 Assignment 4 - Iterative Retrieval for Multi-hop QA.

The harness calls ``run(question, tools)`` once per question and expects
``{"answer": "<short answer>", "citations": ["<doc_id>", ...]}`` (only the first 3 doc_ids are scored).

Strategy: a decompose-then-chain agent (IRCoT / ReAct style) sized for 5 searches + 6 LLM calls.

  1. plan    One LLM call splits the question into Q1 and Q2, where Q2 refers to Q1's answer as "#1".
  2. loop    Search, then ``read`` the current hop: the LLM names the passage that answers the
             sub-question and the short answer. Hop 1's answer (the bridge entity) is substituted into
             Q2, so every hop-2 query depends on what hop 1 retrieved. Searches are cheap and LLM calls
             scarce, so each hop issues two complementary queries. A failed read triggers a deeper,
             reformulated search, and the retry asks for a best guess instead of "unknown".
  3. answer  Hop 2's answer, citing the bridge passage and the answer passage (gold evidence for a
             2-hop question is exactly two paragraphs).

Typical cost: 4 searches and 3 LLM calls (at most 5 and 6). Prompts stay under the 3,000-token limit.
"""

import re
import string

from a4 import BudgetExhausted, LLMError, PromptTooLong, Passage

# ------------------------------------------------------------------------------------------------------
# Tunable constants
# ------------------------------------------------------------------------------------------------------
K_SUBQ = 10                # passages for a sub-question query
K_COMPANION = 10           # passages for the companion query (original question / bridge entity)
K_DEEP = 10                # passages when re-searching a sub-question after a failed read
MAX_PROMPT_PASSAGES = 15  # passages shown to the LLM in one prompt
PASSAGE_CHARS = 1400       # long paragraphs are cut to this many characters in prompts
PROMPT_CHAR_BUDGET = 9500  # ~2,500 tokens at ~3.6 chars/token: margin under the 3,000-token limit
MIN_SECONDS_FOR_RETRY = 15 # do not start an optional retry with less wall-clock time than this
UNKNOWN = "unknown"
RRF_C = 3
HOP2_EXTRA = lambda notes: []


# ======================================================================================================
# Notes: everything the agent has gathered so far for one question
# ======================================================================================================
class Notes:
    """Passages retrieved so far, which hop's query found them, and the state of the two-hop chain."""

    def __init__(self, question: str):
        self.passages: dict[str, Passage] = {}  # doc_id -> passage, de-duplicated
        self.queries: list[str] = []            # queries issued, in order
        self.hits: list[list[str]] = []         # doc_ids returned by each query, in rank order
        self.hop_of_query: list[int] = []       # which hop (0 or 1) each query served
        self.thoughts: list[str] = []           # short log of decisions (handy when debugging)
        self.subqs = [question, question]       # Q1 and Q2 ("#1" = answer of Q1); replaced by plan()
        self.hop = 0                            # hop currently being resolved
        self.attempts = [0, 0]                  # failed reads per hop
        self.answers = ["", ""]                 # answer of each hop ("" = not resolved yet)
        self.answer_docs = [None, None]         # doc_id supporting each hop's answer

    def add(self, query: str, passages: list[Passage]) -> None:
        self.queries.append(query)
        self.hop_of_query.append(self.hop)
        self.hits.append([p.doc_id for p in passages])
        for p in passages:
            self.passages.setdefault(p.doc_id, p)

    def ranked(self, hop: int) -> list[Passage]:
        """All passages, those retrieved for ``hop`` first. Within a group, the result lists of the
        different queries are interleaved by rank (round-robin), so every query contributes its best."""
        order: list[str] = []
        for group in (hop, 1 - hop):
            score: dict[str, float] = {}
            for h, q_hop in zip(self.hits, self.hop_of_query):
                if q_hop == group:
                    for rank, d in enumerate(h):
                        score[d] = score.get(d, 0.0) + 1.0 / (RRF_C + rank + 1)
            order += [d for d in sorted(score, key=lambda d: -score[d]) if d not in order]
        return [self.passages[d] for d in order]

    def current_question(self) -> str:
        """The sub-question of the current hop, with hop 1's answer filled into Q2."""
        q = self.subqs[self.hop]
        return q.replace("#1", self.answers[0]) if self.hop == 1 else q


# ======================================================================================================
# Small text utilities (no LLM)
# ======================================================================================================
def norm(s: str) -> str:
    """Lower-case, drop punctuation and articles, collapse spaces (SQuAD-style normalisation)."""
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def mentions(p: Passage, phrase: str) -> bool:
    """True if the passage (title or text) contains ``phrase`` after normalisation."""
    phrase = norm(phrase)
    return bool(phrase) and f" {phrase} " in f" {norm(p.title + ' ' + p.text)} "


def is_unknown(ans: str) -> bool:
    a = norm(ans)
    return not a or a in {UNKNOWN, "none", "n/a", "na", "no answer"} or a.startswith(
        ("unknown", "not mentioned", "not stated", "not given", "not specified", "passages do not",
         "passage does not", "there is no", "cannot be determined", "no information"))


def clean_answer(text: str) -> str:
    """Strip prefixes, quotes, markdown, passage references and trailing punctuation from an answer."""
    line = text.strip().split("\n")[0].strip()
    line = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", line, flags=re.I)
    line = re.sub(r"^the answer is\s+", "", line, flags=re.I)
    line = re.sub(r"\s*\[\d+\]\s*", " ", line)          # stray passage references like "[3]"
    line = re.sub(r"\s*\((passage|see)[^)]*\)", "", line, flags=re.I)
    return line.strip(" .\"'*`").strip()


def call_llm(tools, build_prompt, passages: list[Passage], max_tokens: int) -> tuple[str, list[Passage]]:
    """Call the LLM with as many of ``passages`` as fit; shrink and retry on PromptTooLong.

    ``build_prompt(block)`` receives the numbered passage block. Returns (output, passages shown).
    """
    shown, size = [], 0
    for p in passages[:MAX_PROMPT_PASSAGES]:  # greedy packing by an approximate character budget
        size += min(len(p.text), PASSAGE_CHARS) + len(p.title) + 8
        if shown and size > PROMPT_CHAR_BUDGET:
            break
        shown.append(p)
    while True:
        block = "\n".join(f"[{i + 1}] {p.title}: {p.text[:PASSAGE_CHARS]}" for i, p in enumerate(shown))
        try:
            return tools.llm(build_prompt(block), max_tokens=max_tokens), shown
        except PromptTooLong:  # not counted against the budget: drop the lowest-ranked passages
            if len(shown) <= 1:
                raise
            shown = shown[: max(1, len(shown) * 2 // 3)]


# ======================================================================================================
# Plan: split the question into two single-hop sub-questions (1 LLM call)
# ======================================================================================================
PLAN_PROMPT = """Split the question into two simple questions. The second question must refer to the answer \
of the first one as #1.

Question: Who is the spouse of the Green performer?
Q1: Who is the performer of Green?
Q2: Who is the spouse of #1?

Question: When was the institute that owned The Collegian founded?
Q1: Which institute owned The Collegian?
Q2: When was #1 founded?

Question: What county contains the city where Walter Robinson was born?
Q1: In which city was Walter Robinson born?
Q2: Which county contains #1?

Question: Who founded the record label of the performer of Me and My Monkey?
Q1: Who is the performer of Me and My Monkey?
Q2: Who founded the record label of #1?

Question: {question}
"""


def plan(question: str, notes: Notes, tools) -> None:
    out = tools.llm(PLAN_PROMPT.format(question=question), max_tokens=80)
    q1 = re.search(r"Q1\s*:\s*(.+)", out)
    q2 = re.search(r"Q2\s*:\s*(.+)", out)
    if q1 and q2 and "#1" in q2.group(1):
        notes.subqs = [q1.group(1).strip(), q2.group(1).strip()]
    notes.thoughts.append(f"plan: Q1={notes.subqs[0]!r} Q2={notes.subqs[1]!r}")


# ======================================================================================================
# Read: try to answer the current hop from the passages (1 LLM call)
# ======================================================================================================
READ_PROMPT = """Answer the question using the passages below.
{context}
Passages:
{passages}

Question: {subq}
{instruction}
Reply in exactly this format:
Passage: <the number of the passage that answers the question>
Answer: <the shortest exact answer span: a name, place, date or number - not a sentence>"""

STRICT = 'If no passage answers the question, reply "Passage: none" and "Answer: unknown".'
GUESS = "If the passages are not explicit, use your own knowledge and give your best guess; never reply unknown."


def read(question: str, notes: Notes, tools) -> None:
    """Ask the LLM to answer the current hop. On success, store the answer and its supporting passage and
    move on to the next hop; on failure, count the attempt so that the next round searches deeper and the
    next read asks for a best guess."""
    hop = notes.hop
    subq = notes.current_question()
    context = f"(This is one step towards answering: {question})" if hop == 0 else \
        f"(Main question: {question} We already know that #1 = {notes.answers[0]}.)"
    last_chance = notes.attempts[hop] >= 1 or tools.budget_left()["llm_calls"] <= 2
    candidates = notes.ranked(hop)
    out, shown = call_llm(
        tools,
        lambda block: READ_PROMPT.format(context=context, passages=block, subq=subq,
                                         instruction=GUESS if last_chance else STRICT),
        candidates, max_tokens=48)
    ans_m = re.search(r"Answer\s*:\s*(.+)", out)
    ans = clean_answer(ans_m.group(1) if ans_m else out)
    notes.thoughts.append(f"read hop {hop + 1}: {subq!r} -> {ans!r}")
    if is_unknown(ans) or (hop == 1 and norm(ans) == norm(notes.answers[0])):
        notes.attempts[hop] += 1
        return
    num = re.search(r"Passage\s*:\s*\[?(\d+)", out)
    cited = shown[int(num.group(1)) - 1] if num and 1 <= int(num.group(1)) <= len(shown) else None
    notes.answers[hop] = ans
    notes.answer_docs[hop] = supporting_doc(ans, cited, candidates, avoid=notes.answer_docs[0])
    notes.hop = 1


def supporting_doc(ans: str, cited, candidates: list[Passage], avoid=None):
    """The doc_id that supports ``ans``: the LLM's cited passage if it really mentions the answer, else the
    best-ranked passage that does (preferring one other than the other hop's passage), else the cite."""
    if cited is not None and mentions(cited, ans) and cited.doc_id != avoid:
        return cited.doc_id
    containing = [p for p in candidates if mentions(p, ans)]
    for p in containing:
        if p.doc_id != avoid:
            return p.doc_id
    if containing:
        return containing[0].doc_id
    return cited.doc_id if cited is not None else None


# ======================================================================================================
# (a) next_query
# ======================================================================================================
def next_query(question: str, notes: Notes, tools) -> list[tuple[str, int]]:
    """Return the next searches as (query, k) pairs, based on what the reads have established so far.

    * Hop 2 just started: Q2 with the bridge entity filled in ("Who is the spouse of Steve Hillage?"),
                          plus the bridge entity on its own, which retrieves the bridge's own article
                          (where facts such as spouse, birthplace or founding date usually live).
    * A read just failed: the current sub-question plus the title of the best-ranked retrieved passage not
                          yet used in a query (a candidate bridge / topic entity), searched deeper (k=10).
                          Each retry takes the next title, so a query is never repeated verbatim.
    """
    subq = notes.current_question()
    if notes.hop == 1 and notes.attempts[1] == 0:
        if subq in notes.queries:  # planning failed (Q2 is the original question): anchor on the bridge
            subq = f"{notes.answers[0]} {subq}"
        return [(subq, K_SUBQ), (f"{notes.answers[0]} {question}", K_COMPANION)] + HOP2_EXTRA(notes)
    used = " ".join(notes.queries)
    for p in notes.ranked(notes.hop):
        if p.title not in used:
            return [(f"{p.title} {subq}", K_DEEP)]
    return [(f"{subq} {question}", K_DEEP)]


def first_queries(question: str, notes: Notes) -> list[tuple[str, int]]:
    """Hop 1: the planned sub-question Q1, plus the original question, whose extra context words often
    rank the bridge passage when Q1 alone is too short for BM25."""
    if norm(notes.subqs[0]) == norm(question):  # planning failed: a single, deeper search
        return [(question, K_DEEP)]
    return [(notes.subqs[0], K_SUBQ), (question, K_COMPANION)]


# ======================================================================================================
# (b) should_stop
# ======================================================================================================
def should_stop(question: str, notes: Notes, tools) -> bool:
    """Stop once hop 2 is answered, or when another read cannot be afforded: one LLM call is always kept
    in reserve for the final answer, and no retry is started with too little wall-clock time left."""
    if notes.answers[1]:
        return True
    left = tools.budget_left()
    return left["llm_calls"] <= 1 or left["seconds"] < MIN_SECONDS_FOR_RETRY


# ======================================================================================================
# (c) answer
# ======================================================================================================
FINAL_PROMPT = """Answer the question using the passages below and your own knowledge.
{hint}
Passages:
{passages}

Question: {question}
Reply with the shortest exact answer only (a name, place, date or number), never "unknown".
Answer:"""


def answer(question: str, notes: Notes, tools) -> dict:
    """Return hop 2's answer if the chain was completed; otherwise make one forced best-guess call over all
    evidence (with the bridge entity as a hint). Cite the bridge passage and the answer passage."""
    final = notes.answers[1]
    if not final:
        hint = f"Hint: {notes.subqs[0]} {notes.answers[0]}" if notes.answers[0] else ""
        try:
            out, shown = call_llm(tools, lambda block: FINAL_PROMPT.format(hint=hint, passages=block,
                                                                           question=question),
                                  notes.ranked(1), max_tokens=32)
            final = clean_answer(out)
            notes.answer_docs[1] = supporting_doc(final, None, shown, avoid=notes.answer_docs[0])
        except (BudgetExhausted, PromptTooLong, LLMError):
            final = notes.answers[0]  # nothing better left: the bridge entity is our only candidate
    return {"answer": final, "citations": citations(notes)}


def citations(notes: Notes) -> list[str]:
    """Bridge passage + answer passage (the two gold paragraphs of a 2-hop question). A missing slot is
    filled with the top-ranked passage of that hop, so we cite two documents whenever we have them."""
    cits = [d for d in notes.answer_docs if d]
    for hop in (0, 1):
        if len(cits) >= 2:
            break
        for p in notes.ranked(hop):
            if p.doc_id not in cits:
                cits.append(p.doc_id)
                break
    return list(dict.fromkeys(cits))[:2]


# ======================================================================================================
# Entry point (called by the harness)
# ======================================================================================================
def run(question: str, tools) -> dict:
    notes = Notes(question)
    try:
        plan(question, notes, tools)
        searches = first_queries(question, notes)
        while True:
            for query, k in searches[: tools.budget_left()["search_calls"]]:
                notes.add(query, tools.search(query, k=k))
            read(question, notes, tools)
            if should_stop(question, notes, tools):
                break
            searches = next_query(question, notes, tools)
    except (BudgetExhausted, PromptTooLong, LLMError):
        pass  # out of budget (or the server failed): answer with what we have
    try:
        return answer(question, notes, tools)
    except Exception:  # never crash: a crash scores 0 on the question
        top = list(notes.passages.values())[:2]
        return {"answer": notes.answers[1] or notes.answers[0] or (top[0].title if top else ""),
                "citations": [p.doc_id for p in top]}
