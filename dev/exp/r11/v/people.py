"""COL 764 Assignment 4: an iterative retrieval agent for 2-hop questions.

The harness calls run(question, tools) once per question. It must return
{"answer": "<short answer>", "citations": ["<doc_id>", ...]}.

Budget per question: 5 searches and 6 LLM calls.

Steps:
  1. plan:   the LLM splits the question into Q1 and Q2. Q2 uses "#1" for Q1's answer.
  2. hop 1:  search for Q1 and read the passages. Q1's answer is the "bridge".
  3. hop 2:  put the bridge into Q2, search again and read again.
             If a read fails, search again. If hop 1 keeps failing, the LLM guesses the bridge.
  4. answer: return hop 2's answer and cite two passages (one for the bridge, one for the answer).
             If hop 2 found nothing, the LLM answers from memory, with the bridge as a hint.
"""

import re
import string

from a4 import BudgetExhausted, LLMError, PromptTooLong, Passage

# Settings
K = 10                     # passages per search (the maximum allowed)
READ_PASSAGES = (10, 12)   # passages shown to the LLM in hop 1 and hop 2
PASSAGE_CHARS = 2000       # no passage is longer, so passages are shown whole
RRF_C = 3                  # reciprocal rank fusion: score = sum of 1 / (RRF_C + rank)
PROMPT_CHAR_BUDGET = 11000 # about 2,750 tokens, under the 3,000-token limit
MIN_SECONDS_FOR_RETRY = 15 # do not start another read with less time left
UNKNOWN = "unknown"
PLAN_MAX_OVERLAP = 0.9     # a Q1 that repeats this much of the question is a bad plan
W_PREFER, W_CITED = 2, 1   # citation weights: right kind of passage, chosen by the LLM
W_WORDS = 1                # citation weight for each sub-question word in the passage
MAX_ANSWER_WORDS = 10      # a longer reply is a sentence, not an answer


class Notes:
    """Everything found so far for one question."""

    def __init__(self, question: str):
        self.passages: dict[str, Passage] = {}  # doc_id -> passage
        self.queries: list[str] = []            # queries sent, in order
        self.hits: list[list[str]] = []         # doc_ids returned by each query, best first
        self.hop_of_query: list[int] = []       # hop (0 or 1) of each query
        self.subqs = [question, question]       # Q1 and Q2 (plan() replaces them)
        self.hop = 0                            # current hop
        self.attempts = [0, 0]                  # failed reads per hop
        self.answers = ["", ""]                 # answer of each hop ("" = not found yet)
        self.rejected: list[str] = []           # hop-2 answers we rejected, kept as fallbacks
        self.cited: list = [None, None]         # passage the LLM chose for each hop's answer
        self.shown: set[str] = set()            # doc_ids already shown in this hop
        self.memory_bridge = False              # True if the bridge came from the LLM's memory

    def add(self, query: str, passages: list[Passage]) -> None:
        self.queries.append(query)
        self.hop_of_query.append(self.hop)
        self.hits.append([p.doc_id for p in passages])
        for p in passages:
            self.passages.setdefault(p.doc_id, p)

    def ranked(self, hop: int) -> list[Passage]:
        """All passages, this hop's first. Each hop's result lists are merged with
        reciprocal rank fusion, so a passage that several queries rank high comes first."""
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
        """The question for this hop. In hop 2, "#1" is replaced by the bridge."""
        q = self.subqs[self.hop]
        return q.replace("#1", self.answers[0]) if self.hop == 1 else q


# Text helpers

def norm(s: str) -> str:
    """Lower-case, remove punctuation and a/an/the, and collapse spaces."""
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def mentions(p: Passage, phrase: str) -> bool:
    """True if the passage contains the phrase. "/" counts as a space ("Town/County")."""
    phrase = norm(phrase)
    return bool(phrase) and f" {phrase} " in f" {norm((p.title + ' ' + p.text).replace('/', ' '))} "


def is_unknown(ans: str) -> bool:
    """True if the reply is empty, says "unknown", or is a whole sentence."""
    a = norm(ans)
    return not a or len(a.split()) > MAX_ANSWER_WORDS or a in {UNKNOWN, "none", "n/a", "na", "no answer"} \
        or a.startswith(("unknown", "not mentioned", "not stated", "not given", "not specified", "not provided",
                         "passage", "there is no", "cannot be determined", "no information", "none of"))


def clean_answer(text: str) -> str:
    """Keep only the answer: drop "Answer:", quotes, passage numbers and a trailing acronym."""
    line = text.strip().split("\n")[0].strip()
    line = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", line, flags=re.I)
    line = re.sub(r"^the answer is\s+", "", line, flags=re.I)
    line = re.sub(r"\s*\[\d+\]\s*", " ", line)             # "[3]"
    line = re.sub(r"\s*\((passage|see)[^)]*\)", "", line, flags=re.I)
    line = re.sub(r"\s*\([A-Z][A-Z.&]{1,9}\)$", "", line)  # "Example Standards Bureau (ESB)"
    return line.strip(" .\"'*`").strip()


def call_llm(tools, build_prompt, passages: list[Passage], max_tokens: int, limit: int):
    """Call the LLM with as many of the first `limit` passages as fit.
    Returns (reply, passages shown)."""
    shown, size = [], 0
    for p in passages[:limit]:
        size += min(len(p.text), PASSAGE_CHARS) + len(p.title) + 8
        if shown and size > PROMPT_CHAR_BUDGET:
            break
        shown.append(p)
    while True:
        block = "\n".join(f"[{i + 1}] {p.title}: {p.text[:PASSAGE_CHARS]}" for i, p in enumerate(shown))
        try:
            return tools.llm(build_prompt(block), max_tokens=max_tokens), shown
        except PromptTooLong:  # not counted as a call: drop some passages and try again
            if len(shown) <= 1:
                raise
            shown = shown[: max(1, len(shown) * 3 // 4)]


# Plan: split the question into Q1 and Q2

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

# Second try when the first plan is bad: the model first names the hidden entity.
# The examples are made up.
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
    """Return (Q1, Q2), or None if the plan is bad: Q2 has no "#1", Q1 has "#1",
    Q1 repeats the whole question, or the model copied a prompt example."""
    q1 = re.search(r"Q1\s*:\s*(.+)", out)
    q2 = re.search(r"Q2\s*:\s*(.+)", out)
    if not (q1 and q2 and "#1" in q2.group(1)):
        return None
    q1, q2 = q1.group(1).strip(), q2.group(1).strip()
    if "#1" in q1:
        return None
    asked = content_words(question)
    if asked and len(content_words(q1) & asked) >= PLAN_MAX_OVERLAP * len(asked):
        return None
    if copied_example(q1, question) or copied_example(q2, question):
        return None
    return q1, q2


def copied_example(sub: str, question: str) -> bool:
    """True if a sub-question is a prompt example that uses words the question does not have."""
    examples = {norm(m) for prompt in (PLAN_PROMPT, PLAN_RETRY_PROMPT, PLAN_ENTITY_PROMPT)
                for m in re.findall(r"Q[12]: (.+)", prompt.split("{question}")[0])}
    return norm(sub) in examples and not content_words(sub.replace("#1", "")) <= content_words(question)


def plan(question: str, notes: Notes, tools) -> None:
    """Ask for a plan; if it is bad, ask once more with another prompt.
    If both are bad, both hops use the original question."""
    first = None
    prompts = [PLAN_PROMPT, PLAN_RETRY_PROMPT]
    for k in range(2):
        out = tools.llm(prompts[k].format(question=question), max_tokens=80)
        q1 = re.search(r"Q1\s*:\s*(.+)", out)
        if q1 and NOT_AN_ENTITY.match(q1.group(1)):
            # Q1 asks for a date or number, so the two hops were merged.
            # Never use this plan; retry with the entity prompt.
            prompts[1] = PLAN_ENTITY_PROMPT
            continue
        parsed = parse_plan(out, question)
        if parsed and not wrong_place_type(question, parsed[0]):
            notes.subqs = list(parsed)
            return
        first = first or parsed
    if first:  # the second plan was no better: keep the first
        notes.subqs = list(first)


# The bridge is a person, place, group or work, never a date, number or reason.
# A Q1 that asks "when" or "how many" is retried with PLAN_ENTITY_PROMPT.
NOT_AN_ENTITY = re.compile(r"\s*(when|why|how (many|much|long|old)|what year|which year|in what year|in which year|until|"
                           r"on (which|what) dates?|what date|in (what|which) century)\b", re.I)

PLAN_ENTITY_PROMPT = """Split the question into two simple questions. The second question must refer to the answer \
of the first one as #1. The first question asks only for the hidden entity (a person, place, group or work), \
never for a date, a number or a reason.

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

Question: When did the trams stop running in the city where Clara Voss died?
Q1: In which city did Clara Voss die?
Q2: When did the trams stop running in #1?

Question: How many times did the team Owen Tallis plays for win the league?
Q1: Which team does Owen Tallis play for?
Q2: How many times did #1 win the league?

Question: Westbrook, in the state where Fennel Ridge is set, is part of which county?
Q1: In which state is Fennel Ridge set?
Q2: Westbrook, in #1, is part of which county?

Question: {question}
"""


PLACE_TYPES = r"\b(country|state|county|province|city|town|region|district|continent|parish)\b"


def wrong_place_type(question: str, q1: str) -> bool:
    """True if Q1 adds a kind of place the question never names ("birthplace of X" ->
    "which country"), or already asks for the place type the whole question wants."""
    in_q1 = {w.lower() for w in re.findall(PLACE_TYPES, q1, re.I)}
    in_q = {w.lower() for w in re.findall(PLACE_TYPES, question, re.I)}
    target = re.match(r"\s*(?:in\s+)?(?:what|which)\s+(?:other\s+)?(\w+)", question, re.I)
    return bool(in_q1 - in_q) or bool(target and target.group(1).lower() in in_q1 and
                                      len(re.findall(PLACE_TYPES, question, re.I)) == 1)


# Read: the LLM answers the current hop from the passages

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
    """Ask the LLM for this hop's answer. A good answer is saved and we move to the next hop.
    Otherwise the attempt counts as failed, and the next round searches again."""
    hop = notes.hop
    subq = notes.current_question()
    context = f"(This is one step towards answering: {question})" if hop == 0 else \
        f"(Main question: {question} We already know that #1 = {notes.answers[0]}.)"
    last_chance = notes.attempts[hop] >= 1 or tools.budget_left()["llm_calls"] <= 2
    ranked = notes.ranked(hop)
    if hop == 0:
        # The article about the entity named in the question usually has the bridge.
        # Put it first, because the model reads the first passages best.
        ranked = [p for p in ranked if title_in(p, question)] + [p for p in ranked if not title_in(p, question)]
    candidates = retry_order(ranked, notes.shown)
    out, shown = call_llm(
        tools,
        lambda block: READ_PROMPT.format(context=context, passages=block, subq=subq,
                                         instruction=GUESS if last_chance else STRICT),
        candidates, max_tokens=48, limit=READ_PASSAGES[hop])
    notes.shown.update(p.doc_id for p in shown)
    ans_m = re.search(r"Answer\s*:\s*(.+)", out)
    ans = clean_answer(ans_m.group(1) if ans_m else out)
    # Reject a hop-2 answer of the wrong type, or one that just repeats the bridge.
    wrong_type = hop == 1 and not fits_type(question, ans)
    if is_unknown(ans) or wrong_type or (hop == 1 and norm(ans) == norm(notes.answers[0])):
        notes.attempts[hop] += 1
        if not is_unknown(ans):
            notes.rejected.append(ans)
        return
    num = re.search(r"Passage\s*:\s*\[?(\d+)", out)
    cited = shown[int(num.group(1)) - 1] if num and 1 <= int(num.group(1)) <= len(shown) else None
    if hop == 0 and notes.attempts[0] == 0 and cited is not None and shown and cited.doc_id != shown[0].doc_id \
            and not title_in(cited, question):
        ans, cited = recheck_weak_bridge(question, notes, tools, context, subq, ans, cited)
    if hop == 1 and notes.attempts[1] == 0 and cited is not None and not mentions(cited, ans) \
            and any(mentions(p, ans) for p in shown) \
            and tools.budget_left()["llm_calls"] >= 2 and len(shown) >= 2:
        # The cited passage lacks the answer but another passage has it: the model mixed up
        # its evidence. Get a second opinion; if it only repeats the answer, search again.
        # Needs 2 calls left, or the second opinion would just return the same answer.
        checked, checked_cited = second_opinion(question, notes, tools, context, shown, ans, cited, last_chance)
        if norm(checked) == norm(ans):
            notes.attempts[1] += 1
            notes.rejected.append(ans)
            return
        ans, cited = checked, checked_cited
    elif hop == 1 and notes.attempts[1] == 0:
        ans, cited = second_opinion(question, notes, tools, context, shown, ans, cited, last_chance)
    if hop == 1:
        ans, cited = memory_check(question, notes, tools, context, subq, question, ans, cited)
    notes.answers[hop] = ans
    notes.cited[hop] = cited
    notes.hop = 1
    notes.shown = set()


JUDGE_PROMPT = """{context}
Question: {subq}

Two answers were given, each with the passage it came from.
Answer A: {a}
Passage A: {pa}
Answer B: {b}
Passage B: {pb}

Which answer is correct? Reply with only the letter A or B."""


CLOSED_PROMPT = """Answer the question from your own knowledge. Reply with only the shortest exact answer
(a name, place, date or number), or "unknown" if you do not know.
Question: {q}
Answer:"""


BRIDGE_GUESS_PROMPT = """This is one step towards answering: {question}
Question: {q}
Answer from your own knowledge with the shortest exact answer (a name or place). Give your best guess; never
reply unknown.
Answer:"""


# The model knows newer facts than the dataset, so do not ask its memory about things that change.
TIME_SENSITIVE = re.compile(r"\b(last|latest|most recent|current|currently|now|today|recent)\b", re.I)


def memory_check(question: str, notes: Notes, tools, context: str, asked: str, judge_q: str, ans: str, cited):
    """Ask the model to answer from memory, without passages. If its answer is different and a
    passage contains it, the judge picks one (in hop 2 a spare search may look for that passage).
    Our answer is A and wins on any doubt."""
    left = tools.budget_left()
    if left["llm_calls"] < 2 or left["seconds"] < 20 or TIME_SENSITIVE.search(question):
        return ans, cited
    try:
        out = tools.llm(CLOSED_PROMPT.format(q=asked), max_tokens=24)
        other = clean_answer(re.sub(r"^\s*answer\s*:\s*", "", out, flags=re.I))
        if is_unknown(other) or norm(other) in norm(ans) or norm(ans) in norm(other):
            return ans, cited
        if notes.hop == 1 and (norm(other) == norm(notes.answers[0]) or not fits_type(question, other)):
            return ans, cited
        pool = notes.ranked(notes.hop)
        p2 = next((p for p in pool if mentions(p, other)), None)  # memory answers need a passage too
        if p2 is None and notes.hop == 1 and notes.answers[0] and tools.budget_left()["search_calls"] >= 1:
            # No passage has the memory answer: use the spare search on "bridge + answer".
            found = tools.search(f"{notes.answers[0]} {other}", k=K)
            notes.add(f"{notes.answers[0]} {other}", found)
            p2 = next((p for p in found if mentions(p, other)), None)
        p1 = cited if cited is not None and mentions(cited, ans) else \
            next((p for p in pool if mentions(p, ans)), cited)
        if p1 is None or p2 is None:
            return ans, cited
        verdict = tools.llm(JUDGE_PROMPT.format(context=context.strip(), subq=judge_q, a=ans, b=other,
                                                pa=f"{p1.title}: {p1.text[:PASSAGE_CHARS]}",
                                                pb=f"{p2.title}: {p2.text[:PASSAGE_CHARS]}"), max_tokens=4)
        if re.match(r"\W*(answer\s*)?b\b", verdict.strip(), re.I):
            return other, p2
    except (BudgetExhausted, PromptTooLong, LLMError):
        pass
    return ans, cited


def recheck_weak_bridge(question: str, notes: Notes, tools, context: str, subq: str, ans: str, cited):
    """A bridge read from a passage that is neither first nor the article named in the question
    is often wrong. Search again with Q1 plus a guessed answer sentence (Query2doc), read hop 1
    again, and take the new answer if it is a real one."""
    left = tools.budget_left()
    if left["llm_calls"] < 4 or left["search_calls"] < 3 or left["seconds"] < 30:
        return ans, cited
    passage = hypothetical_passage(subq, tools)
    if not passage:
        return ans, cited
    query = f"{keywords(subq)} {re.split(r'(?<=[.!?])\s', passage)[0]}"
    try:
        notes.add(query, tools.search(query, k=K))
        ranked = notes.ranked(0)
        ranked = [p for p in ranked if title_in(p, question)] + [p for p in ranked if not title_in(p, question)]
        out, shown = call_llm(tools, lambda block: READ_PROMPT.format(context=context, passages=block, subq=subq,
                                                                      instruction=STRICT),
                              ranked, max_tokens=48, limit=READ_PASSAGES[0])
    except (BudgetExhausted, PromptTooLong, LLMError):
        return ans, cited
    m = re.search(r"Answer\s*:\s*(.+)", out)
    new = clean_answer(m.group(1) if m else out)
    if is_unknown(new):
        return ans, cited
    num = re.search(r"Passage\s*:\s*\[?(\d+)", out)
    return new, shown[int(num.group(1)) - 1] if num and 1 <= int(num.group(1)) <= len(shown) else None


def second_opinion(question: str, notes: Notes, tools, context: str, shown: list[Passage],
                   ans: str, cited, last_chance: bool):
    """Check a hop-2 answer with a second read: the same passages in reverse order (the model is
    sensitive to order), asked with the original question (the plan's Q2 can be badly written).
    If the answers differ, the judge picks one. The first answer is A and wins on any doubt.
    Returns (answer, cited passage)."""
    left = tools.budget_left()
    if left["llm_calls"] < 2 or left["seconds"] < 25 or len(shown) < 2:
        return ans, cited
    try:
        rev = list(reversed(shown))
        block = "\n".join(f"[{i + 1}] {p.title}: {p.text[:PASSAGE_CHARS]}" for i, p in enumerate(rev))
        out = tools.llm(READ_PROMPT.format(context=context, passages=block, subq=question,
                                           instruction=GUESS if last_chance else STRICT), max_tokens=48)
        m = re.search(r"Answer\s*:\s*(.+)", out)
        other = clean_answer(m.group(1) if m else out)
        if is_unknown(other) or norm(other) == norm(ans) or norm(other) == norm(notes.answers[0]) \
                or not fits_type(question, other):
            return ans, cited
        # Not a real disagreement if the second answer only adds words to the first (short
        # answers are preferred), unless the added words are date parts ("1902" -> "June 1902").
        words, more = set(norm(ans).split()), set(norm(other).split())
        if words < more and not all(re.fullmatch(r"\d+|" + MONTHS, w) for w in more - words):
            return ans, cited
        # Also not a real disagreement if the second answer is a piece of the first ("12th" -> "12").
        if norm(other) in norm(ans):
            return ans, cited
        num = re.search(r"Passage\s*:\s*\[?(\d+)", out)
        cited2 = rev[int(num.group(1)) - 1] if num and 1 <= int(num.group(1)) <= len(rev) else None
        p1 = cited if cited is not None and mentions(cited, ans) else \
            next((p for p in shown if mentions(p, ans)), cited)
        p2 = cited2 if cited2 is not None and mentions(cited2, other) else \
            next((p for p in rev if mentions(p, other)), cited2)
        if p1 is None or p2 is None:
            return ans, cited
        verdict = tools.llm(JUDGE_PROMPT.format(context=context.strip(), subq=question, a=ans, b=other,
                                                pa=f"{p1.title}: {p1.text[:PASSAGE_CHARS]}",
                                                pb=f"{p2.title}: {p2.text[:PASSAGE_CHARS]}"), max_tokens=4)
        if re.match(r"\W*(answer\s*)?b\b", verdict.strip(), re.I):
            return other, p2
    except (BudgetExhausted, PromptTooLong, LLMError):
        pass
    return ans, cited


def retry_order(ranked: list[Passage], seen: set[str]) -> list[Passage]:
    """On a retry, mix new passages with ones already shown. The best shown passage stays first."""
    if not seen:
        return ranked
    new = [p for p in ranked if p.doc_id not in seen]
    old = [p for p in ranked if p.doc_id in seen]
    mixed = [p for pair in zip(old, new) for p in pair]
    n = min(len(old), len(new))
    return mixed + old[n:] + new[n:]


def overlap(p: Passage, words: set[str]) -> int:
    """How many of the words appear in the passage."""
    return len(words & set(norm(p.title + " " + p.text).split()))


def supporting_doc(ans: str, cited, candidates: list[Passage], avoid=None, prefer=None, words=frozenset()):
    """Pick the passage to cite for an answer. It must contain the answer (and not be `avoid` if
    possible). Points for passing `prefer`, for being the LLM's choice, and for each of `words`
    it contains; ties go to the best ranked. If no passage has the answer, use the LLM's choice."""
    containing = [p for p in candidates if mentions(p, ans)]
    containing = [p for p in containing if p.doc_id != avoid] or containing
    if not containing:
        return cited.doc_id if cited is not None else None

    def merit(item):
        rank, p = item
        is_cited = cited is not None and p.doc_id == cited.doc_id
        return (W_PREFER * bool(prefer and prefer(p)) + W_CITED * is_cited + W_WORDS * overlap(p, words), -rank)

    return max(enumerate(containing), key=merit)[1].doc_id


def title_in(p: Passage, question: str) -> bool:
    """True if the passage title (without a bracket at the end) appears in the question."""
    title = norm(re.sub(r"\s*\([^)]*\)\s*$", "", p.title))
    return bool(title) and f" {title} " in f" {norm(question)} "


# (a) next_query

PSEUDO_PROMPT = """Write a short Wikipedia-style paragraph (two or three sentences) that answers the question.
Question: {q}
Paragraph:"""


def hypothetical_passage(subq: str, tools) -> str:
    """Ask the LLM to write a passage that might answer the question. Its facts may be wrong,
    but its words help BM25 (Query2doc)."""
    try:
        return " ".join(tools.llm(PSEUDO_PROMPT.format(q=subq), max_tokens=96).split())
    except (BudgetExhausted, PromptTooLong, LLMError):
        return ""


def next_query(question: str, notes: Notes, tools) -> list[tuple[str, int]]:
    """Return the next searches as (query, k) pairs.

    - Hop 2 starts: Q2 with the bridge filled in, and the bridge plus the question.
    - A hop-2 read failed: the bridge alone (finds its own article), or for "when"
      questions the bridge plus a guessed answer sentence.
    - A hop-1 read failed: Q1 plus a guessed answer sentence, or the bridge from memory.
    - Otherwise: the title of the best passage not searched yet, plus the question's keywords.
    """
    subq = notes.current_question()
    bridge = notes.answers[0]
    if notes.hop == 1 and notes.attempts[1] == 0:
        if "#1" not in notes.subqs[1]:  # no plan: Q2 is the whole question, so add the bridge
            return [(f"{bridge} {keywords(question)}", K), (bridge, K)]
        return [(keywords(subq), K), (f"{bridge} {keywords(question)}", K)]
    if notes.hop == 1 and notes.attempts[1] == 1 and tools.budget_left()["llm_calls"] >= 3 \
            and re.match(r"\s*(when|what year|in what year|which year|what date)\b", question, re.I):
        # "When" question: the bridge's own article rarely has the date, so search with the
        # bridge plus a guessed answer sentence instead.
        passage = hypothetical_passage(subq, tools)
        if passage:
            first = re.split(r"(?<=[.!?])\s", passage)[0]  # first sentence only, so the bridge counts most
            return [(f"{bridge} {bridge} {first}", K)]
    if notes.hop == 1 and bridge not in notes.queries:
        return [(bridge, K)]
    if notes.hop == 0 and (notes.attempts[0] >= 2 or tools.budget_left()["llm_calls"] < 4) \
            and notes.attempts[0] >= 1 and tools.budget_left()["llm_calls"] >= 2 \
            and tools.budget_left()["search_calls"] >= 1 and tools.budget_left()["seconds"] >= 20:
        # Hop 1 keeps failing. The model often knows the bridge anyway: ask its memory and
        # go on to hop 2, whose search and read still ground the final answer.
        try:
            out = tools.llm(BRIDGE_GUESS_PROMPT.format(question=question, q=notes.subqs[0]), max_tokens=24)
        except (BudgetExhausted, PromptTooLong, LLMError):
            out = ""
        guess = clean_answer(re.sub(r"^\s*answer\s*:\s*", "", out, flags=re.I))
        asked = content_words(question)
        real_split = len(content_words(notes.subqs[0]) & asked) <= 0.75 * len(asked)  # Q1 is not the whole question
        if usable(guess) and real_split and "#1" in notes.subqs[1] and not NOT_AN_ENTITY.match(notes.subqs[0]):
            notes.answers[0], notes.cited[0] = guess, None
            notes.memory_bridge = True
            notes.hop, notes.shown = 1, set()
            return next_query(question, notes, tools)
    if notes.hop == 0 and notes.attempts[0] == 1 and tools.budget_left()["llm_calls"] >= 4:
        # Hop 1 found nothing: search with Q1 plus the first sentence of a guessed answer.
        passage = hypothetical_passage(subq, tools)
        if passage:
            return [(f"{keywords(subq)} {re.split(r'(?<=[.!?])\s', passage)[0]}", K)]
    used = " ".join(notes.queries)
    for p in notes.ranked(notes.hop):
        if p.title not in used:
            return [(f"{p.title} {keywords(question)}", K)]
    return [(f"{subq} {question}", K)]


def first_queries(question: str, notes: Notes) -> list[tuple[str, int]]:
    """Hop 1 searches Q1 and the whole question (it adds words when Q1 is short)."""
    if norm(notes.subqs[0]) == norm(question):  # no plan: one search
        return [(question, K)]
    return [(notes.subqs[0], K), (question, K)]


QUESTION_WORDS = re.compile(r"\b(what|which|who|whom|whose|when|where|why|how|is|are|was|were|did|does|do)\b",
                            re.I)


def keywords(query: str) -> str:
    """The query without question words and question marks (they do not help BM25)."""
    return " ".join(QUESTION_WORDS.sub(" ", query).replace("?", " ").split())


# (b) should_stop

def should_stop(question: str, notes: Notes, tools) -> bool:
    """Stop when hop 2 has an answer, or when there is no budget for another read
    (one LLM call is kept for answer(), and no read starts when time is short)."""
    if notes.answers[1]:
        return True
    left = tools.budget_left()
    return left["llm_calls"] <= 1 or left["seconds"] < MIN_SECONDS_FOR_RETRY


# (c) answer

FINAL_PROMPT = """Answer the question using the passages below and your own knowledge.
{hint}
Passages:
{passages}

Question: {question}
Reply with the shortest exact answer only (a name, place, date or number), never "unknown".
Answer:"""


MEMORY_FINAL_PROMPT = """This is the last step towards answering: {question}
{hint}
Question: {q}
Answer from your own knowledge with the shortest exact answer (a name, place, date or number). Give your best
guess; never reply unknown.
Answer:"""


def answer(question: str, notes: Notes, tools) -> dict:
    """Return the final answer and two citations.

    If hop 2 has no answer, one more LLM call asks for a best guess (from memory if we know the
    bridge, else from the passages). Fallbacks, in order: a rejected hop-2 answer of the right
    type, the reply itself, the bridge, and a span from the best passage. Then the answer is
    put into a short, standard form."""
    final = notes.answers[1]
    if not final:
        hint = f"Hint: {notes.subqs[0]} {notes.answers[0]}" if notes.answers[0] else ""
        try:
            if notes.answers[0]:
                # A clean question answered from memory beats a guess from noisy passages.
                out = tools.llm(MEMORY_FINAL_PROMPT.format(question=question, hint=hint, q=question), max_tokens=32)
                final = clean_answer(re.sub(r"^\s*answer\s*:\s*", "", out, flags=re.I))
            else:
                out, shown = call_llm(tools, lambda block: FINAL_PROMPT.format(hint=hint, passages=block,
                                                                               question=question),
                                      notes.ranked(1), max_tokens=32, limit=READ_PASSAGES[1])
                final = clean_answer(out)
        except (BudgetExhausted, PromptTooLong, LLMError):
            pass
    typed = [a for a in reversed(notes.rejected) if fits_type(question, a)]
    if typed and not (usable(final) and fits_type(question, final)):
        final = typed[0]
    if not usable(final):
        sentence = final if final and not is_unknown(final) else ""
        final = (notes.rejected or [sentence or notes.answers[0]])[-1]
    if not usable(final) or re.match(r"(not available|not known|no one|nobody)\b", norm(final)):
        final = extract_guess(question, notes) or final  # an empty answer always scores 0
    if re.search(r"\b(what|which)\s+(other\s+)?county\b", question, re.I) and final \
            and not re.search(r"\b(county|parish|borough|shire)\b", final, re.I):
        final_text = f"{final} County"  # the question asks for a county: name it as one
    else:
        final_text = final
    long_form = re.search(r"\b(\w+)\s*-\s*long\b", final_text)
    if long_form and re.match(r"\s*how long\b", question, re.I):  # "a week-long festival" -> "week"
        final_text = long_form.group(1)
    if re.search(r"\bcentur(y|ies)\b", question, re.I):  # write ordinals as digits: "9th", not "ninth"
        for i, w in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
                              "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth "
                              "twentieth".split(), 1):
            suffix = "th" if 10 < i < 14 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
            final_text = re.sub(rf"\b{w}\b", f"{i}{suffix}", final_text, flags=re.I)
    answer_text = final_form(drop_unit(question, expand_acronym(final_text, notes)))
    return {"answer": answer_text, "citations": citations(question, final, notes)}


def extract_guess(question: str, notes: Notes) -> str:
    """Last resort, with no LLM call: from the passage that shares most words with the question,
    take a date (for "when"), a number (for "how many"), or else its title."""
    pool = notes.ranked(1)[:10] if notes.answers[0] else notes.ranked(0)[:10]
    words = content_words(question)
    pool = sorted(pool, key=lambda p: -overlap(p, words))
    q = question.lower().strip()
    if re.match(r"(when|what year|which year|in what year|in which year|what date)\b", q):
        span = re.compile(r"\b(?:(?:\d{1,2}\s+)?(?:" + MONTHS + r")(?:\s+\d{1,2})?,?\s+\d{4}|"
                          r"\d{1,2}(?:st|nd|rd|th)\s+centur(?:y|ies)|1\d{3}s?|20\d{2}s?)\b", re.I)
    elif re.match(r"how (many|much|old|long|tall|high|big)\b", q) or re.search(r"\b(population|number of)\b", q):
        span = re.compile(r"\b\d[\d,.]*\d\b|\b\d\b")
    else:
        span = None
    for p in pool:
        m = span.search(p.text) if span else None
        if m:
            return m.group(0)
        if span is None and not title_in(p, question):
            return re.sub(r"\s*\([^)]*\)\s*$", "", p.title)
    return ""


def expand_acronym(ans: str, notes: Notes) -> str:
    """Spell out a bare acronym ("NFL" -> "National Football League") when a passage writes
    "Full Name (NFL)" and the initials match. A full name is clearer than initials."""
    if not re.fullmatch(r"[A-Z]{2,6}", ans):
        return ans
    for p in notes.passages.values():
        for m in re.finditer(r"((?:[A-Z][\w'&-]*\s+(?:of\s+|and\s+|for\s+|the\s+)?){1,7})\(" + ans + r"\)", p.text):
            words = m.group(1).split()
            while words:  # drop leading words that are not part of the name
                if "".join(w[0] for w in words if w[0].isupper()) == ans:
                    return " ".join(words)
                words = words[1:]
    return ans


def drop_unit(question: str, ans: str) -> str:
    """For "how many X" questions keep only the number ("three goals" -> "three"), if the question
    uses that unit. Also "twelve (12)" -> "twelve"."""
    ans = re.sub(r"^(\S+)\s*\(\d[\d,.]*\)$", r"\1", ans.strip())
    if not re.match(r"how many\b", question.strip(), re.I):
        return ans
    m = re.match(r"^(\d[\d,.]*|(?:" + NUMBER_WORDS + r")(?:[- ](?:" + NUMBER_WORDS + r"))*)\s+(.+)$", ans, re.I)
    stem = lambda text: {w.rstrip("s") for w in norm(text).split()}  # "goal" matches "goals"
    if m and stem(m.group(2)) <= stem(question) | {"time"}:
        return m.group(1)
    return ans


MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
NUMBER_WORDS = (r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\w+teen|twenty|thirty|forty|"
                "fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|dozen|once|twice|single")


def fits_type(question: str, ans: str) -> bool:
    """False if the answer has the wrong type: no county for a county question, no date for a
    "when" question, or no number for a "how many/much/long" question."""
    q, a = question.lower().strip(), ans.lower()
    if re.search(r"\b(what|which)\s+(other\s+)?county\b|\bbirth county\b", q):
        return bool(re.search(r"\b(county|parish|borough|shire)\b", a))
    if re.match(r"(when|what year|which year|in what year|in which year|what date)\b", q):
        return bool(re.search(r"\d|\b(" + MONTHS + r"|century|centuries|decade|era|age|war|period|dynasty|"
                              r"spring|summer|autumn|fall|winter|season)\b", a))
    if re.match(r"how (many|much|long)\b", q):
        units = r"|year|years|month|months|week|weeks|day|days|hour|hours"
        return bool(re.search(r"\d|\b(" + NUMBER_WORDS + units + r")\b", a))
    return True


def final_form(ans: str) -> str:
    """Format the answer for the scorer, which deletes punctuation: put spaces around hyphens
    so "1987-06-30" stays three words. Also trim extra words that a short answer does not need."""
    ans = ans.replace("*", "").strip()
    # "trade guild for weavers" -> "trade guild"; "the Norse people" -> "the Norse"
    head, sep, _ = ans.partition(" for ")
    if sep and len(head.split()) >= 2 and not re.search(r"\d", ans):
        ans = head
    group = re.match(r"(.+?)\s+people$", ans, re.I)
    if False:  # keep "the people" as it is
        ans = group.group(1)
    ans = re.sub(r"^(?:a |an |one )?(\w+)\s*-\s*long$", r"\1", ans, flags=re.I)  # "week-long" -> "week"
    ans = re.sub(r"^1\s+(year|month|week|day|hour|decade|century)$", r"\1", ans, flags=re.I)  # "1 month" -> "month"
    # "Riverton, West Yorkshire" -> "Riverton". Keep dates, numbers, lists and "Washington, D.C.".
    head, sep, tail = ans.partition(", ")
    if sep and re.match(r"[A-Za-z]", tail) and "." not in tail and not re.search(r"\b(and|or)\b|&", ans):
        ans = head
    ans = re.sub(r"\s*[-\u2013\u2014]+\s*", " - ", ans)
    # "about 300" -> "300"
    return re.sub(r"^(over|about|approximately|around|nearly|almost|more than|less than)\s+(?=\d)", "", ans,
                  flags=re.I)


def usable(ans: str) -> bool:
    """True for a short answer (not "unknown" and not a sentence)."""
    return bool(ans) and not is_unknown(ans) and len(ans.split()) <= MAX_ANSWER_WORDS


def named_in(p: Passage, question: str) -> bool:
    """Like title_in, but hyphens count as spaces ("Silver-tail ferry" = "Silver tail ferry").
    Used only to choose citations."""
    dash = lambda s: re.sub(r"[-–—]", " ", s)
    return title_in(p, question) or title_in(Passage(p.doc_id, dash(p.title), p.text, 0.0), dash(question))


def citations(question: str, final: str, notes: Notes) -> list[str]:
    """Cite two passages, chosen after all searches:
    - one with the bridge, preferably the article named in the question;
    - one with the answer, preferably also naming the bridge.
    If one is missing, use the top passage of that hop."""
    bridge = notes.answers[0]
    cits = []
    named = [p for p in notes.ranked(0) if named_in(p, question)]
    if bridge and notes.memory_bridge and named:
        cits.append(named[0].doc_id)  # the bridge came from memory: cite the article the question names
    elif bridge:
        cits.append(supporting_doc(bridge, notes.cited[0], notes.ranked(0),
                                   prefer=lambda p: named_in(p, question), words=content_words(notes.subqs[0])))
    if final and final != bridge:
        cits.append(supporting_doc(final, notes.cited[1], notes.ranked(1), avoid=cits[0] if cits else None,
                                   prefer=lambda p: bool(bridge) and mentions(p, bridge),
                                   words=content_words(notes.subqs[1].replace("#1", "")) | content_words(question)))
    cits = list(dict.fromkeys(d for d in cits if d))
    for hop in (0, 1):
        if len(cits) >= 2:
            break
        pool = notes.ranked(hop)
        if hop == 0:  # for the bridge, try the article named in the question first
            pool = [p for p in pool if named_in(p, question)] + pool
        for p in pool:
            if p.doc_id not in cits:
                cits.append(p.doc_id)
                break
    return cits[:2]


# Entry point, called by the harness

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
        pass  # out of budget or the server failed: answer with what we have
    try:
        return answer(question, notes, tools)
    except Exception:  # never crash: a crash scores 0
        top = list(notes.passages.values())[:2]
        return {"answer": notes.answers[1] or notes.answers[0] or (top[0].title if top else ""),
                "citations": [p.doc_id for p in top]}
