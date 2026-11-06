PATCH = [
    ('''    return {"answer": final_form(final), "citations": citations(question, final, notes)}''',
     '''    return {"answer": final_form(drop_unit(question, final)), "citations": citations(question, final, notes)}


def drop_unit(question: str, ans: str) -> str:
    """For "how many X" questions keep only the number: "one season" -> "one", "5 times" -> "5".
    Only words that the question already uses are dropped. A number in brackets after a number word
    is dropped too: "thirty (30)" -> "thirty"."""
    ans = re.sub(r"^(\\S+)\\s*\\(\\d[\\d,.]*\\)$", r"\\1", ans.strip())
    if not re.match(r"how many\\b", question.strip(), re.I):
        return ans
    m = re.match(r"^(\\d[\\d,.]*|(?:" + NUMBER_WORDS + r")(?:[- ](?:" + NUMBER_WORDS + r"))*)\\s+(.+)$", ans, re.I)
    stem = lambda text: {w.rstrip("s") for w in norm(text).split()}  # "season" matches "seasons"
    if m and stem(m.group(2)) <= stem(question) | {"time"}:
        return m.group(1)
    return ans'''),
]
