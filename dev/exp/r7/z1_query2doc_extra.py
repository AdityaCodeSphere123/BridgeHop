"""COL 764 Assignment 4: iterative retrieval for 2-hop questions.

The harness calls run(question, tools) once per question and expects
{"answer": "<short answer>", "citations": ["<doc_id>", ...]}.

How the agent works (budget: 5 searches and 6 LLM calls per question):
  1. plan:   the LLM splits the question into Q1 and Q2. Q2 uses "#1" for the answer of Q1.
  2. hop 1:  search for Q1, then the LLM reads the passages and gives Q1's answer (the "bridge").
  3. hop 2:  put the bridge into Q2, search again, and let the LLM read the new passages.
             If a read fails, search once more and ask the LLM for its best guess.
  4. answer: return hop 2's answer and cite two passages: one for the bridge, one for the answer.
"""

import re
import string

from a4 import BudgetExhausted, LLMError, PromptTooLong, Passage

# Settings
K = 10                     # passages returned by each search (the maximum allowed)
READ_PASSAGES = (10, 12)   # passages shown to the LLM when reading hop 1 and hop 2
PASSAGE_CHARS = 1200       # long passages are cut to this many characters
RRF_C = 3                  # constant in reciprocal rank fusion: score = sum of 1 / (RRF_C + rank)
PROMPT_CHAR_BUDGET = 9000  # about 2,300 tokens, safely under the 3,000-token prompt limit
MIN_SECONDS_FOR_RETRY = 15 # do not start an extra read with less time than this left
UNKNOWN = "unknown"
PLAN_MAX_OVERLAP = 0.9     # if Q1 repeats this share of the question's words, the plan is not used
W_PREFER, W_CITED = 2, 1   # citation choice: weight for "right kind of passage" and for "the LLM chose it"
W_WORDS = 1                # citation choice: weight for each word of the sub-question found in the passage
MAX_ANSWER_WORDS = 10      # a longer reply is a sentence, not an answer


class Notes:
    """Everything the agent has found so far for one question."""

    def __init__(self, question: str):
        self.passages: dict[str, Passage] = {}  # doc_id -> passage
        self.queries: list[str] = []            # queries sent, in order
        self.hits: list[list[str]] = []         # doc_ids returned by each query, best first
        self.hop_of_query: list[int] = []       # which hop (0 or 1) each query was for
        self.subqs = [question, question]       # Q1 and Q2; plan() replaces them
        self.hop = 0                            # the hop being worked on
        self.attempts = [0, 0]                  # failed reads per hop
        self.answers = ["", ""]                 # answer of each hop ("" = not found yet)
        self.rejected: list[str] = []           # hop-2 answers we did not accept, kept as fallbacks
        self.cited: list = [None, None]         # passage the LLM chose for each hop's answer
        self.shown: set[str] = set()            # doc_ids already shown in this hop's reads

    def add(self, query: str, passages: list[Passage]) -> None:
        self.queries.append(query)
        self.hop_of_query.append(self.hop)
        self.hits.append([p.doc_id for p in passages])
        for p in passages:
            self.passages.setdefault(p.doc_id, p)

    def ranked(self, hop: int) -> list[Passage]:
        """All passages, those found for this hop first. A hop's result lists are merged with
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
        """The question for the current hop. In hop 2, "#1" is replaced by the bridge."""
        q = self.subqs[self.hop]
        return q.replace("#1", self.answers[0]) if self.hop == 1 else q


# Text helpers

def norm(s: str) -> str:
    """Lower-case, remove punctuation and the words a/an/the, and collapse spaces."""
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def mentions(p: Passage, phrase: str) -> bool:
    """True if the passage's title or text contains the phrase."""
    phrase = norm(phrase)
    return bool(phrase) and f" {phrase} " in f" {norm(p.title + ' ' + p.text)} "


def is_unknown(ans: str) -> bool:
    """True if the reply is empty, says it does not know, or is a whole sentence."""
    a = norm(ans)
    return not a or len(a.split()) > MAX_ANSWER_WORDS or a in {UNKNOWN, "none", "n/a", "na", "no answer"} \
        or a.startswith(("unknown", "not mentioned", "not stated", "not given", "not specified", "not provided",
                         "passage", "there is no", "cannot be determined", "no information", "none of"))


def clean_answer(text: str) -> str:
    """Keep only the answer: remove "Answer:", quotes, passage numbers and a trailing acronym."""
    line = text.strip().split("\n")[0].strip()
    line = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", line, flags=re.I)
    line = re.sub(r"^the answer is\s+", "", line, flags=re.I)
    line = re.sub(r"\s*\[\d+\]\s*", " ", line)             # passage numbers like "[3]"
    line = re.sub(r"\s*\((passage|see)[^)]*\)", "", line, flags=re.I)
    line = re.sub(r"\s*\([A-Z][A-Z.&]{1,9}\)$", "", line)  # "Drug Enforcement Administration (DEA)"
    return line.strip(" .\"'*`").strip()


def call_llm(tools, build_prompt, passages: list[Passage], max_tokens: int, limit: int):
    """Call the LLM with as many of the first `limit` passages as fit in the prompt.
    Returns (reply, passages shown). build_prompt(block) gets the numbered passages."""
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
        except PromptTooLong:  # this does not use up a call: drop the last passages and try again
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

# Second try, used when the first plan is not usable. The model first names the hidden entity.
# The examples are made up and show the most common question shapes.
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
    """Return (Q1, Q2), or None if the plan is not usable: Q2 has no "#1", or Q1 just repeats
    the whole question (then hop 1 would try to answer the final question)."""
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
    """Ask for a plan; if it is not usable, ask once more with the second prompt.
    If both fail, both hops keep the original question."""
    for prompt in (PLAN_PROMPT, PLAN_RETRY_PROMPT):
        parsed = parse_plan(tools.llm(prompt.format(question=question), max_tokens=80), question)
        if parsed:
            notes.subqs = list(parsed)
            break


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
    """Ask the LLM for the current hop's answer. If the answer is good, save it and move to the next hop.
    If not, count a failed attempt: the next round searches again and asks for a best guess."""
    hop = notes.hop
    subq = notes.current_question()
    context = f"(This is one step towards answering: {question})" if hop == 0 else \
        f"(Main question: {question} We already know that #1 = {notes.answers[0]}.)"
    last_chance = notes.attempts[hop] >= 1 or tools.budget_left()["llm_calls"] <= 2
    ranked = notes.ranked(hop)
    if hop == 0:
        # The article about the entity named in the question usually holds hop 1's fact.
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
    # A hop-2 answer is not accepted if it has the wrong type for the question or just repeats the bridge.
    wrong_type = hop == 1 and not fits_type(question, ans)
    if is_unknown(ans) or wrong_type or (hop == 1 and norm(ans) == norm(notes.answers[0])):
        notes.attempts[hop] += 1
        if not is_unknown(ans):
            notes.rejected.append(ans)
        return
    num = re.search(r"Passage\s*:\s*\[?(\d+)", out)
    cited = shown[int(num.group(1)) - 1] if num and 1 <= int(num.group(1)) <= len(shown) else None
    if hop == 1 and notes.attempts[1] == 0:
        ans, cited = second_opinion(question, notes, tools, context, subq, shown, ans, cited, last_chance)
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


# The model's own knowledge is newer than the dataset, so it is not asked about things that change over time.
TIME_SENSITIVE = re.compile(r"\b(last|latest|most recent|current|currently|now|today|recent)\b", re.I)


def memory_check(question: str, notes: Notes, tools, context: str, asked: str, judge_q: str, ans: str, cited):
    """A third, independent opinion: the model answers from its own knowledge, without passages. If that
    answer differs from ours and some retrieved passage contains it, the judge chooses between the two,
    each shown with its passage. Our answer is option A and wins on any doubt."""
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
        p2 = next((p for p in pool if mentions(p, other)), None)  # only answers that a passage supports
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


def second_opinion(question: str, notes: Notes, tools, context: str, subq: str, shown: list[Passage],
                   ans: str, cited, last_chance: bool):
    """Check a hop-2 answer with a second read: the same passages in reverse order (the model is sensitive
    to passage order), asked with the original question instead of the plan's Q2 (which can be badly
    written). If the two answers really differ, one more call asks the model to choose between them, each
    shown with the passage it came from. The first answer is option A and is kept whenever there is any
    doubt or no budget. Returns (answer, cited passage)."""
    left = tools.budget_left()
    if left["llm_calls"] < 2 or left["seconds"] < 25 or len(shown) < 2:
        return ans, cited
    try:
        rev = list(reversed(shown))
        block = "\n".join(f"[{i + 1}] {p.title}: {p.text[:PASSAGE_CHARS]}" for i, p in enumerate(rev))
        # The second read answers the original question, so an answer that is wrong only because the
        # plan wrote a bad Q2 can be caught.
        out = tools.llm(READ_PROMPT.format(context=context, passages=block, subq=question,
                                           instruction=GUESS if last_chance else STRICT), max_tokens=48)
        m = re.search(r"Answer\s*:\s*(.+)", out)
        other = clean_answer(m.group(1) if m else out)
        if is_unknown(other) or norm(other) == norm(ans) or norm(other) == norm(notes.answers[0]) \
                or not fits_type(question, other):
            return ans, cited
        # If the second answer only adds words to the first, keep the first: gold answers are short.
        # Added date parts are fine ("1810" -> "October 1810").
        words, more = set(norm(ans).split()), set(norm(other).split())
        if words < more and not all(re.fullmatch(r"\d+|" + MONTHS, w) for w in more - words):
            return ans, cited
        # The second answer is a piece of the first ("34th" -> "34"): not a real disagreement either.
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
    """On a retry, mix passages not shown yet with the ones already shown.
    The best passage already shown stays first."""
    if not seen:
        return ranked
    new = [p for p in ranked if p.doc_id not in seen]
    old = [p for p in ranked if p.doc_id in seen]
    mixed = [p for pair in zip(old, new) for p in pair]
    n = min(len(old), len(new))
    return mixed + old[n:] + new[n:]


def overlap(p: Passage, words: set[str]) -> int:
    """How many of the words appear in the passage (title or text)."""
    return len(words & set(norm(p.title + " " + p.text).split()))


def supporting_doc(ans: str, cited, candidates: list[Passage], avoid=None, prefer=None, words=frozenset()):
    """Pick the passage to cite for an answer: one that contains the answer (not `avoid` if possible).
    Prefer one that passes `prefer`, the one the LLM chose and one that shares many `words` with the
    sub-question (several paragraphs of the same article often contain the answer); then the best ranked.
    If no passage contains the answer, use the LLM's choice."""
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
    """True if the passage's title (without a bracket at the end) appears in the question."""
    title = norm(re.sub(r"\s*\([^)]*\)\s*$", "", p.title))
    return bool(title) and f" {title} " in f" {norm(question)} "


# (a) next_query

PSEUDO_PROMPT = """Write a short Wikipedia-style paragraph (two or three sentences) that answers the question.
Question: {q}
Paragraph:"""


def hypothetical_passage(subq: str, tools) -> str:
    """Query2doc / HyDE: the LLM writes a passage that might answer the question. Its facts may be wrong,
    but its words (names, terms) help BM25 find the real passage that uses other words than the question."""
    try:
        return " ".join(tools.llm(PSEUDO_PROMPT.format(q=subq), max_tokens=96).split())
    except (BudgetExhausted, PromptTooLong, LLMError):
        return ""


def next_query(question: str, notes: Notes, tools) -> list[tuple[str, int]]:
    """Return the next searches as (query, k) pairs.

    - Hop 2 starts: Q2 with the bridge filled in, and the bridge followed by the question.
      Question words (what, when, ...) are removed because they do not help BM25.
    - A hop-2 read failed: search for the bridge alone, which finds the bridge's own article.
    - Otherwise: the title of the best passage not used in a query yet, plus the question's keywords.
    """
    subq = notes.current_question()
    bridge = notes.answers[0]
    if notes.hop == 1 and notes.attempts[1] == 0:
        if "#1" not in notes.subqs[1]:  # no usable plan: Q2 is the whole question, so add the bridge
            return [(f"{bridge} {keywords(question)}", K), (bridge, K)]
        queries = [(keywords(subq), K), (f"{bridge} {keywords(question)}", K)]
        if tools.budget_left()["llm_calls"] >= 4:  # Query2doc: also search with a hypothetical passage
            passage = hypothetical_passage(subq, tools)
            if passage:
                queries.append((f"{bridge} {passage}", K))
        return queries
    if notes.hop == 1 and bridge not in notes.queries:
        return [(bridge, K)]
    used = " ".join(notes.queries)
    for p in notes.ranked(notes.hop):
        if p.title not in used:
            return [(f"{p.title} {keywords(question)}", K)]
    return [(f"{subq} {question}", K)]


def first_queries(question: str, notes: Notes) -> list[tuple[str, int]]:
    """Hop 1 searches Q1 and the whole question. The whole question adds useful words when Q1 is short."""
    if norm(notes.subqs[0]) == norm(question):  # no usable plan: one search
        return [(question, K)]
    return [(notes.subqs[0], K), (question, K)]


QUESTION_WORDS = re.compile(r"\b(what|which|who|whom|whose|when|where|why|how|is|are|was|were|did|does|do)\b",
                            re.I)


def keywords(query: str) -> str:
    """The query without question words and question marks."""
    return " ".join(QUESTION_WORDS.sub(" ", query).replace("?", " ").split())


# (b) should_stop

def should_stop(question: str, notes: Notes, tools) -> bool:
    """Stop when hop 2 has an answer, or when there is no budget for another read:
    one LLM call is kept for answer(), and no extra read is started when time is short."""
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


def answer(question: str, notes: Notes, tools) -> dict:
    """Return the final answer and two citations.

    If hop 2 has no answer, one more LLM call asks for a best guess. An answer of the right type is
    preferred. If the reply is still a sentence or "unknown", fall back to a rejected hop-2 answer,
    then to the sentence itself, then to the bridge."""
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
    typed = [a for a in reversed(notes.rejected) if fits_type(question, a)]
    if typed and not (usable(final) and fits_type(question, final)):
        final = typed[0]
    if not usable(final):
        sentence = final if final and not is_unknown(final) else ""
        final = (notes.rejected or [sentence or notes.answers[0]])[-1]
    return {"answer": final_form(drop_unit(question, final)), "citations": citations(question, final, notes)}


def drop_unit(question: str, ans: str) -> str:
    """For "how many X" questions keep only the number: "one season" -> "one", "5 times" -> "5".
    Only words that the question already uses are dropped. A number in brackets after a number word
    is dropped too: "thirty (30)" -> "thirty"."""
    ans = re.sub(r"^(\S+)\s*\(\d[\d,.]*\)$", r"\1", ans.strip())
    if not re.match(r"how many\b", question.strip(), re.I):
        return ans
    m = re.match(r"^(\d[\d,.]*|(?:" + NUMBER_WORDS + r")(?:[- ](?:" + NUMBER_WORDS + r"))*)\s+(.+)$", ans, re.I)
    stem = lambda text: {w.rstrip("s") for w in norm(text).split()}  # "season" matches "seasons"
    if m and stem(m.group(2)) <= stem(question) | {"time"}:
        return m.group(1)
    return ans


MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
NUMBER_WORDS = (r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\w+teen|twenty|thirty|forty|"
                "fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|dozen|once|twice|single")


def fits_type(question: str, ans: str) -> bool:
    """False if the answer clearly has the wrong type:
    - "what/which county" or "birth county" question, but no county in the answer;
    - "when" or "what year" question, but no date, number or time period;
    - "how many/much/long" question, but no number.
    Every other answer fits."""
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
    """Format the answer for the scorer. The scorer deletes punctuation, so "2001-02-13" would become
    one word. Put spaces around hyphens so each part counts. Also drop "over", "about", ... before a
    number, because gold answers are short."""
    ans = ans.replace("*", "").strip()
    # "Leyton, East London" -> "Leyton": gold answers rarely add a region after a comma.
    # Dates and numbers ("May 5, 1940", "8,491,079"), lists and abbreviations ("Washington, D.C.") stay.
    head, sep, tail = ans.partition(", ")
    if sep and re.match(r"[A-Za-z]", tail) and "." not in tail and not re.search(r"\b(and|or)\b|&", ans):
        ans = head
    ans = re.sub(r"\s*[-\u2013\u2014]+\s*", " - ", ans)
    return re.sub(r"^(over|about|approximately|around|nearly|almost|more than|less than)\s+(?=\d)", "", ans,
                  flags=re.I)


def usable(ans: str) -> bool:
    """True for a short answer (not "unknown" and not a sentence)."""
    return bool(ans) and not is_unknown(ans) and len(ans.split()) <= MAX_ANSWER_WORDS


def citations(question: str, final: str, notes: Notes) -> list[str]:
    """Cite two passages, chosen after all searches:
    - the bridge passage: contains the bridge, preferably the article named in the question;
    - the answer passage: contains the answer, preferably also the bridge.
    If one is missing, use the top passage of that hop."""
    bridge = notes.answers[0]
    cits = []
    if bridge:
        cits.append(supporting_doc(bridge, notes.cited[0], notes.ranked(0),
                                   prefer=lambda p: title_in(p, question), words=content_words(notes.subqs[0])))
    if final and final != bridge:
        cits.append(supporting_doc(final, notes.cited[1], notes.ranked(1), avoid=cits[0] if cits else None,
                                   prefer=lambda p: bool(bridge) and mentions(p, bridge),
                                   words=content_words(notes.subqs[1].replace("#1", "")) | content_words(question)))
    cits = list(dict.fromkeys(d for d in cits if d))
    for hop in (0, 1):
        if len(cits) >= 2:
            break
        pool = notes.ranked(hop)
        if hop == 0:  # for hop 1, try the article named in the question first
            pool = [p for p in pool if title_in(p, question)] + pool
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
    except Exception:  # never crash, because a crash scores 0
        top = list(notes.passages.values())[:2]
        return {"answer": notes.answers[1] or notes.answers[0] or (top[0].title if top else ""),
                "citations": [p.doc_id for p in top]}
