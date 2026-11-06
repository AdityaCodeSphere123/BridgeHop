"""COL 764 Assignment 4 - Iterative Retrieval for Multi-hop QA.

The harness calls ``run(question, tools)`` once per question and expects
``{"answer": "<short answer>", "citations": ["<doc_id>", ...]}`` (only the first 3 doc_ids are scored).

Strategy: a decompose-then-chain agent (IRCoT / ReAct style) sized for 5 searches + 6 LLM calls.

  1. plan    One LLM call splits the question into Q1 and Q2, where Q2 refers to Q1's answer as "#1"
             (a second, firmer call only if the first reply cannot be parsed).
  2. loop    Search, then ``read`` the current hop: the LLM names the passage that answers the
             sub-question and the short answer. Hop 1's answer (the bridge entity) is substituted into
             Q2, so every hop-2 query depends on what hop 1 retrieved. Each hop issues two complementary
             k=10 queries whose result lists are fused by reciprocal rank. A failed read triggers a new
             search; the retry read shows the passages not seen yet first and asks for a best guess
             instead of "unknown".
  3. answer  Hop 2's answer (never a refusal or a sentence when a real candidate exists), citing the
             bridge passage and the answer passage (gold evidence for a 2-hop question is exactly two
             paragraphs). Citations are chosen after all searches, from every retrieved passage.

Typical cost: 4 searches and 3-4 LLM calls (at most 5 and 6). Prompts stay under the 3,000-token limit.
"""

import re
import string

from a4 import BudgetExhausted, LLMError, PromptTooLong, Passage

# ------------------------------------------------------------------------------------------------------
# Tunable constants
# ------------------------------------------------------------------------------------------------------
K = 10                     # passages per search (the maximum): searches are scarce, so each returns all it can
READ_PASSAGES = (10, 12)   # passages shown when reading hop 1 / hop 2 (hop 2 fuses two broad searches)
PASSAGE_CHARS = 1200       # long paragraphs are cut to this many characters in prompts
RRF_C = 3                  # reciprocal rank fusion: score = sum over the hop's queries of 1 / (RRF_C + rank)
PROMPT_CHAR_BUDGET = 9000  # ~2,300 tokens at the measured ~0.25 tokens/char: margin under the 3,000 limit
MIN_SECONDS_FOR_RETRY = 15 # do not start an optional retry with less wall-clock time than this
UNKNOWN = "unknown"
PLAN_MAX_OVERLAP = 0.9     # a Q1 containing this share of the question's content words just restates it
W_PREFER, W_CITED = 2, 1   # citation choice: weight of "fits the hop" vs "the LLM named it"
MAX_ANSWER_WORDS = 10      # a longer final output is a sentence or a refusal, not an answer


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
        self.rejected: list[str] = []           # hop-2 answers rejected because they repeat the bridge
        self.cited: list = [None, None]         # passage the LLM named for each hop's answer
        self.shown: set[str] = set()            # doc_ids already shown in a read of the current hop

    def add(self, query: str, passages: list[Passage]) -> None:
        self.queries.append(query)
        self.hop_of_query.append(self.hop)
        self.hits.append([p.doc_id for p in passages])
        for p in passages:
            self.passages.setdefault(p.doc_id, p)

    def ranked(self, hop: int) -> list[Passage]:
        """All passages, those retrieved for ``hop`` first. Within a hop, the result lists of its queries are
        fused by reciprocal rank, so a passage that several queries rank highly comes first."""
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
    """True for empty answers, refusals and sentence-like replies, which cannot be a short answer."""
    a = norm(ans)
    return not a or len(a.split()) > MAX_ANSWER_WORDS or a in {UNKNOWN, "none", "n/a", "na", "no answer"} \
        or a.startswith(("unknown", "not mentioned", "not stated", "not given", "not specified", "not provided",
                         "passage", "there is no", "cannot be determined", "no information", "none of"))


def clean_answer(text: str) -> str:
    """Strip prefixes, quotes, markdown, passage references and trailing punctuation from an answer."""
    line = text.strip().split("\n")[0].strip()
    line = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", line, flags=re.I)
    line = re.sub(r"^the answer is\s+", "", line, flags=re.I)
    line = re.sub(r"\s*\[\d+\]\s*", " ", line)          # stray passage references like "[3]"
    line = re.sub(r"\s*\((passage|see)[^)]*\)", "", line, flags=re.I)
    line = re.sub(r"\s*\([A-Z][A-Z.&]{1,9}\)$", "", line)  # trailing acronym: "Drug ... (DEA)"
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

# Used when the first plan is unusable (no "#1" in Q2, or Q1 just restates the question). Naming the hidden
# entity first makes the model separate the two lookups; the examples cover the most common nested shapes.
PLAN_RETRY_PROMPT = """Each question hides one unnamed entity, described by a phrase like "the birthplace \
of X" or "X's team". Name that phrase, then write Q1, asking only for the hidden entity, and Q2, the rest of \
the question, with #1 for Q1's answer. Q1 must never ask the whole question.

Question: How many titles did Rafe Holloway's club win?
Hidden entity: Rafe Holloway's club
Q1: Which club did Rafe Holloway play for?
Q2: How many titles did #1 win?

Question: When did trams stop running in the birthplace of Edith Marlow?
Hidden entity: the birthplace of Edith Marlow
Q1: Where was Edith Marlow born?
Q2: When did trams stop running in #1?

Question: When did the river on which Kessler Dam is located flood?
Hidden entity: the river on which Kessler Dam is located
Q1: On which river is Kessler Dam located?
Q2: When did #1 flood?

Question: In which county of the state where The Quiet Harvest is set is Brookfield?
Hidden entity: the state where The Quiet Harvest is set
Q1: In which state is The Quiet Harvest set?
Q2: In which county of #1 is Brookfield?

Question: When did the network that aired Sunday Garden stop showing cartoons?
Hidden entity: the network that aired Sunday Garden
Q1: Which network aired Sunday Garden?
Q2: When did #1 stop showing cartoons?

Question: How many hits did the original singer of Paper Lantern have?
Hidden entity: the original singer of Paper Lantern
Q1: Who was the original singer of Paper Lantern?
Q2: How many hits did #1 have?

Question: {question}
"""

PLAN_STOPWORDS = frozenset("a an the of in on at to for by with from and or is was were are did do does "
                           "what which who whom whose when where why how".split())


def content_words(text: str) -> set[str]:
    return {w for w in norm(text).split() if w not in PLAN_STOPWORDS}


def parse_plan(out: str, question: str):
    """(Q1, Q2) if the reply is a usable plan, else None. A plan is unusable when Q2 does not refer to #1, or
    when Q1 restates (nearly) the whole question - then hop 1 would try to answer the final question."""
    q1 = re.search(r"Q1\s*:\s*(.+)", out)
    q2 = re.search(r"Q2\s*:\s*(.+)", out)
    if not (q1 and q2 and "#1" in q2.group(1)):
        return None
    q1, q2 = q1.group(1).strip(), q2.group(1).strip()
    asked = content_words(question)
    if asked and len(content_words(q1) & asked) >= PLAN_MAX_OVERLAP * len(asked):
        return None
    return q1, q2


def plan(question: str, notes: Notes, tools) -> None:
    """Split the question into Q1 and Q2 ("#1" = Q1's answer). If the first reply is unusable, ask once more
    with a prompt that makes the model name the hidden entity first; if that fails too, both hops use the
    original question."""
    for prompt in (PLAN_PROMPT, PLAN_RETRY_PROMPT):
        parsed = parse_plan(tools.llm(prompt.format(question=question), max_tokens=80), question)
        if parsed:
            notes.subqs = list(parsed)
            break
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
Evidence: <the few words of that passage that state the answer>
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
    ranked = notes.ranked(hop)
    if hop == 0:  # the article on the question's topic entity usually states hop 1's fact: put it first,
        # since the model reads the first passages most reliably
        ranked = [p for p in ranked if title_in(p, question)] + [p for p in ranked if not title_in(p, question)]
    candidates = retry_order(ranked, notes.shown)
    out, shown = call_llm(
        tools,
        lambda block: READ_PROMPT.format(context=context, passages=block, subq=subq,
                                         instruction=GUESS if last_chance else STRICT),
        candidates, max_tokens=96, limit=READ_PASSAGES[hop])  # room for the evidence quote
    notes.shown.update(p.doc_id for p in shown)
    ans_m = re.search(r"Answer\s*:\s*(.+)", out)
    ans = clean_answer(ans_m.group(1) if ans_m else out)
    notes.thoughts.append(f"read hop {hop + 1}: {subq!r} -> {ans!r}")
    if is_unknown(ans) or (hop == 1 and norm(ans) == norm(notes.answers[0])):
        notes.attempts[hop] += 1
        if not is_unknown(ans):
            notes.rejected.append(ans)
        return
    num = re.search(r"Passage\s*:\s*\[?(\d+)", out)
    cited = shown[int(num.group(1)) - 1] if num and 1 <= int(num.group(1)) <= len(shown) else None
    notes.answers[hop] = ans
    notes.cited[hop] = cited
    notes.hop = 1
    notes.shown = set()


def retry_order(ranked: list[Passage], seen: set[str]) -> list[Passage]:
    """Passage order for a read. On a retry the passages already shown did not yield an answer, so passages
    not seen yet are interleaved with them; the best already-seen passage stays first, because the model
    reads the first passages most reliably."""
    if not seen:
        return ranked
    new = [p for p in ranked if p.doc_id not in seen]
    old = [p for p in ranked if p.doc_id in seen]
    mixed = [p for pair in zip(old, new) for p in pair]
    n = min(len(old), len(new))
    return mixed + old[n:] + new[n:]


def supporting_doc(ans: str, cited, candidates: list[Passage], avoid=None, prefer=None):
    """The doc_id that supports ``ans``: among the passages that mention the answer (other than the other
    hop's passage, when possible), the best by (fits the hop's ``prefer`` test, is the LLM's cited passage,
    rank). Falls back to the cited passage when no retrieved passage mentions the answer."""
    containing = [p for p in candidates if mentions(p, ans)]
    containing = [p for p in containing if p.doc_id != avoid] or containing
    if not containing:
        return cited.doc_id if cited is not None else None

    def merit(item):
        rank, p = item
        is_cited = cited is not None and p.doc_id == cited.doc_id
        return (W_PREFER * bool(prefer and prefer(p)) + W_CITED * is_cited, -rank)

    return max(enumerate(containing), key=merit)[1].doc_id


def title_in(p: Passage, question: str) -> bool:
    """True if the passage's title (without a "(disambiguation)" suffix) is named in the question: such a
    passage is the article about the question's topic entity, where hop 1's fact usually lives."""
    title = norm(re.sub(r"\s*\([^)]*\)\s*$", "", p.title))
    return bool(title) and f" {title} " in f" {norm(question)} "


# ======================================================================================================
# (a) next_query
# ======================================================================================================
def next_query(question: str, notes: Notes, tools) -> list[tuple[str, int]]:
    """Return the next searches as (query, k) pairs, based on what the reads have established so far.

    * Hop 2 just started: Q2 with the bridge entity filled in ("spouse Steve Hillage"), plus the bridge
                          entity followed by the original question, whose extra words match paragraphs that
                          state the fact without repeating Q2's wording. Question words are dropped: they
                          match nothing useful in BM25.
    * Hop 2 read failed:  the bridge entity on its own, which retrieves the bridge's own article.
    * Otherwise:          the title of the best-ranked retrieved passage not yet used in a query (a candidate
                          bridge / topic entity) plus the question's keywords. Each retry takes the next
                          title, so a query is never repeated verbatim.
    """
    subq = notes.current_question()
    bridge = notes.answers[0]
    if notes.hop == 1 and notes.attempts[1] == 0:
        if "#1" not in notes.subqs[1]:  # planning failed: Q2 is the original question, so anchor on the bridge
            return [(f"{bridge} {keywords(question)}", K), (bridge, K)]
        return [(keywords(subq), K), (f"{bridge} {keywords(question)}", K)]
    if notes.hop == 1 and bridge not in notes.queries:
        return [(bridge, K)]
    used = " ".join(notes.queries)
    for p in notes.ranked(notes.hop):
        if p.title not in used:
            return [(f"{p.title} {keywords(question)}", K)]
    return [(f"{subq} {question}", K)]


def first_queries(question: str, notes: Notes) -> list[tuple[str, int]]:
    """Hop 1: the planned sub-question Q1, plus the original question, whose extra context words often
    rank the bridge passage when Q1 alone is too short for BM25."""
    if norm(notes.subqs[0]) == norm(question):  # planning failed: a single search
        return [(question, K)]
    return [(notes.subqs[0], K), (question, K)]


QUESTION_WORDS = re.compile(r"\b(what|which|who|whom|whose|when|where|why|how|is|are|was|were|did|does|do)\b",
                            re.I)


def keywords(query: str) -> str:
    """The query without question words and punctuation (BM25 drops only a few stopwords itself)."""
    return " ".join(QUESTION_WORDS.sub(" ", query).replace("?", " ").split())


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
    evidence (with the bridge entity as a hint). Cite the bridge passage and the answer passage.

    A sentence or a refusal is never preferred over a real candidate: the fallbacks are a hop-2 answer that
    was rejected for repeating the bridge, then the sentence itself (it may still overlap the gold answer),
    then the bridge entity."""
    final = notes.answers[1]
    if not final:
        hint = f"Hint: {notes.subqs[0]} {notes.answers[0]}" if notes.answers[0] else ""
        try:
            out, shown = call_llm(tools, lambda block: FINAL_PROMPT.format(hint=hint, passages=block,
                                                                           question=question),
                                  notes.ranked(1), max_tokens=32, limit=READ_PASSAGES[1])
            final = clean_answer(out)
        except (BudgetExhausted, PromptTooLong, LLMError):
            pass
    if not usable(final):
        sentence = final if final and not is_unknown(final) else ""
        final = (notes.rejected or [sentence or notes.answers[0]])[-1]
    return {"answer": final, "citations": citations(question, final, notes)}


def usable(ans: str) -> bool:
    """A short answer, not a refusal or a sentence."""
    return bool(ans) and not is_unknown(ans) and len(ans.split()) <= MAX_ANSWER_WORDS


def citations(question: str, final: str, notes: Notes) -> list[str]:
    """Bridge passage + answer passage (the two gold paragraphs of a 2-hop question), chosen once all
    searches are done, so a passage retrieved after a hop was read can still be cited.

    * bridge passage: mentions the bridge entity; preferably the article on the question's topic entity.
    * answer passage: mentions the final answer; preferably also the bridge entity.
    A slot with no candidate is filled with the top-ranked passage of that hop, so we cite two documents
    whenever we have them."""
    bridge = notes.answers[0]
    cits = []
    if bridge:
        cits.append(supporting_doc(bridge, notes.cited[0], notes.ranked(0),
                                   prefer=lambda p: title_in(p, question)))
    if final and final != bridge:
        cits.append(supporting_doc(final, notes.cited[1], notes.ranked(1), avoid=cits[0] if cits else None,
                                   prefer=lambda p: bool(bridge) and mentions(p, bridge)))
    cits = list(dict.fromkeys(d for d in cits if d))
    for hop in (0, 1):
        if len(cits) >= 2:
            break
        pool = notes.ranked(hop)
        if hop == 0:  # the topic entity's own article is the likeliest hop-1 paragraph
            pool = [p for p in pool if title_in(p, question)] + pool
        for p in pool:
            if p.doc_id not in cits:
                cits.append(p.doc_id)
                break
    return cits[:2]


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
