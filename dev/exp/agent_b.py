"""COL 764 Assignment 4 - Iterative Retrieval for Multi-hop QA.

The harness calls ``run(question, tools)`` once per question and expects
``{"answer": "<short answer>", "citations": ["<doc_id>", ...]}`` (only the first 3 doc_ids are scored).

Strategy: a decompose-then-chain agent (IRCoT / ReAct style) sized for 5 searches + 6 LLM calls.

  1. plan    One LLM call splits the question into Q1 and Q2, where Q2 refers to Q1's answer as "#1".
  2. loop    Search, then ``read`` the current hop: the LLM names the passage that answers the
             sub-question and the short answer. Hop 1's answer (the bridge entity) is substituted into
             Q2, so every hop-2 query depends on what hop 1 retrieved. Each hop issues two complementary
             queries whose result lists are fused (reciprocal rank fusion) before reading. A failed read
             triggers a reformulated search, and the retry asks for a best guess instead of "unknown".
  3. answer  Hop 2's answer, citing the bridge passage and the answer passage (gold evidence for a
             2-hop question is exactly two paragraphs).
"""

import re
import string

from a4 import BudgetExhausted, LLMError, PromptTooLong, Passage

# ------------------------------------------------------------------------------------------------------
# Tunable constants
# ------------------------------------------------------------------------------------------------------
K = 10                     # passages per search (the maximum)
READ_PASSAGES = (10, 15)   # passages shown when reading hop 1 / hop 2
PASSAGE_CHARS = 1200       # long paragraphs are cut to this many characters in prompts
PROMPT_CHAR_BUDGET = 9000  # ~2,600 tokens: margin under the 3,000-token limit
RRF_C = 3                  # reciprocal rank fusion constant: score = sum 1 / (RRF_C + rank)
MIN_SECONDS_FOR_RETRY = 15 # do not start an optional retry with less wall-clock time than this
MAX_ANSWER_WORDS = 10      # longer outputs are sentences, not answers
UNKNOWN = "unknown"
READ_STYLE = "plain"       # "plain": passage + answer; "evidence": also quote the supporting sentence


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
        self.guesses: list[str] = []            # answers rejected or produced as best guesses

    def add(self, query: str, passages: list[Passage]) -> None:
        self.queries.append(query)
        self.hop_of_query.append(self.hop)
        self.hits.append([p.doc_id for p in passages])
        for p in passages:
            self.passages.setdefault(p.doc_id, p)

    def ranked(self, hop: int) -> list[Passage]:
        """All passages, those retrieved for ``hop`` first. Within a hop, the result lists of its queries
        are fused by reciprocal rank, so a passage found by several queries rises to the top."""
        order: list[str] = []
        for group in (hop, 1 - hop):
            score: dict[str, float] = {}
            for hits, q_hop in zip(self.hits, self.hop_of_query):
                if q_hop == group:
                    for rank, d in enumerate(hits):
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
    """True for empty answers, refusals and sentence-like outputs that cannot be a short answer."""
    a = norm(ans)
    return not a or len(a.split()) > MAX_ANSWER_WORDS or a in {UNKNOWN, "none", "n/a", "na", "no answer"} \
        or a.startswith(("unknown", "not mentioned", "not stated", "not given", "not specified", "passages",
                         "passage", "there is no", "cannot be determined", "no information", "none of"))


def clean_answer(text: str) -> str:
    """Strip prefixes, quotes, markdown, passage references, a trailing parenthetical (an acronym or a
    note) and trailing punctuation from an answer."""
    line = text.strip().split("\n")[0].strip()
    line = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", line, flags=re.I)
    line = re.sub(r"^the answer is\s+", "", line, flags=re.I)
    line = re.sub(r"\s*\[\d+\]\s*", " ", line)          # stray passage references like "[3]"
    line = re.sub(r"\s*\([^)]*\)\s*$", "", line)       # "Drug Enforcement Administration (DEA)"
    return line.strip(" .\"'*`").strip()


def call_llm(tools, build_prompt, passages: list[Passage], max_tokens: int, limit: int):
    """Call the LLM with as many of the first ``limit`` passages as fit; shrink and retry on PromptTooLong.

    ``build_prompt(block)`` receives the numbered passage block. Returns (output, passages shown).
    """
    shown, size = [], 0
    for p in passages[:limit]:  # greedy packing by an approximate character budget
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
            shown = shown[: max(1, len(shown) * 3 // 4)]


# ======================================================================================================
# Plan: split the question into two single-hop sub-questions (1 LLM call)
# ======================================================================================================
PLAN_PROMPT = """Every question below needs two lookups: first find a hidden entity, then ask about it. \
Rewrite the question as two simple questions. Q1 finds the hidden entity. Q2 asks the rest of the \
question about it and refers to Q1's answer as #1. Always write both lines.

Question: Who is the spouse of the Green performer?
Q1: Who is the performer of Green?
Q2: Who is the spouse of #1?

Question: When did Mark Ellison's team last win the championship?
Q1: Which team did Mark Ellison play for?
Q2: When did #1 last win the championship?

Question: What is the population of the city where WQRT is licensed?
Q1: In which city is WQRT licensed?
Q2: What is the population of #1?

Question: Which county borders the county that contains Lake Arden?
Q1: Which county contains Lake Arden?
Q2: Which county borders #1?

Question: In what year did the company that published Star Runner release its first console?
Q1: Which company published Star Runner?
Q2: In what year did #1 release its first console?

Question: What language was spoken in the region where the oldest recorded strike took place?
Q1: In which region did the oldest recorded strike take place?
Q2: What language was spoken in #1?

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
{evidence}Answer: <the shortest exact answer span: a name, place, date or number - not a sentence>"""

EVIDENCE_LINE = "Evidence: <the words of that passage that state the answer>\n"
STRICT = 'If no passage answers the question, reply "Passage: none" and "Answer: unknown".'
GUESS = "If the passages are not explicit, use your own knowledge and give your best guess; never reply unknown."


def read(question: str, notes: Notes, tools) -> None:
    """Ask the LLM to answer the current hop. On success, store the answer and its supporting passage and
    move on to the next hop; on failure, count the attempt so that the next round searches again and the
    next read asks for a best guess."""
    hop = notes.hop
    subq = notes.current_question()
    context = f"(This is one step towards answering: {question})" if hop == 0 else \
        f"(Main question: {question} We already know that #1 = {notes.answers[0]}.)"
    last_chance = notes.attempts[hop] >= 1 or tools.budget_left()["llm_calls"] <= 2
    candidates = notes.ranked(hop)
    evidence = EVIDENCE_LINE if READ_STYLE == "evidence" else ""
    out, shown = call_llm(
        tools,
        lambda block: READ_PROMPT.format(context=context, passages=block, subq=subq, evidence=evidence,
                                         instruction=GUESS if last_chance else STRICT),
        candidates, max_tokens=96 if evidence else 48, limit=READ_PASSAGES[hop])
    ans_m = re.search(r"Answer\s*:\s*(.+)", out)
    ans = clean_answer(ans_m.group(1)) if ans_m else ""
    notes.thoughts.append(f"read hop {hop + 1}: {subq!r} -> {ans!r}")
    if is_unknown(ans) or (hop == 1 and norm(ans) == norm(notes.answers[0])):
        notes.attempts[hop] += 1
        if not is_unknown(ans):
            notes.guesses.append(ans)
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
                          plus the bridge entity followed by the original question, whose extra words
                          match paragraphs that state the fact without repeating Q2's wording.
    * A read just failed: hop 2 - the bridge entity on its own, which retrieves the bridge's own article;
                          otherwise the current sub-question plus the title of the best-ranked retrieved
                          passage not yet used in a query (a candidate bridge / topic entity).
    """
    subq = notes.current_question()
    bridge = notes.answers[0]
    if notes.hop == 1 and notes.attempts[1] == 0:
        if subq in notes.queries:  # planning failed (Q2 is the original question): anchor on the bridge
            subq = f"{bridge} {subq}"
        return [(subq, K), (f"{bridge} {question}", K)]
    if notes.hop == 1 and bridge not in notes.queries:
        return [(bridge, K)]
    used = " ".join(notes.queries)
    for p in notes.ranked(notes.hop):
        if p.title not in used:
            return [(f"{p.title} {subq}", K)]
    return [(f"{subq} {question}", K)]


def first_queries(question: str, notes: Notes) -> list[tuple[str, int]]:
    """Hop 1: the planned sub-question Q1, plus the original question, whose extra context words often
    rank the bridge passage when Q1 alone is too short for BM25."""
    if norm(notes.subqs[0]) == norm(question):  # planning failed: a single search
        return [(question, K)]
    return [(notes.subqs[0], K), (question, K)]


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
                                  notes.ranked(1), max_tokens=32, limit=READ_PASSAGES[1])
            final = clean_answer(out)
            if is_unknown(final):
                final = ""
            else:
                notes.answer_docs[1] = supporting_doc(final, None, shown, avoid=notes.answer_docs[0])
        except (BudgetExhausted, PromptTooLong, LLMError):
            pass
    if not final:  # nothing usable: a rejected guess, else the bridge entity, is our only candidate
        final = (notes.guesses or [notes.answers[0]])[-1]
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
