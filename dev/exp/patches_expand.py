OLD = '''    return {"answer": final_form(drop_unit(question, final)), "citations": citations(question, final, notes)}'''
NEW = '''    cits = citations(question, final, notes)
    return {"answer": final_form(drop_unit(question, full_name(final, notes, cits))), "citations": cits}


def full_name(ans: str, notes: Notes, cits: list[str]) -> str:
    """If a cited passage writes the answer as part of a longer capitalized name ("CNBC" in "CNBC Asia"),
    return the longer name. Only words directly next to the answer, with no punctuation between, are added."""
    core = re.sub(r"^(the|a|an)\\s+", "", ans.strip(), flags=re.I)
    if not core or not core[0].isupper() or re.search(r"\\d", core):
        return ans
    for doc_id in cits:
        p = notes.passages.get(doc_id)
        if p is None:
            continue
        cap = r"(?:[A-Z][\\w'\\-]*)"
        m = re.search(r"((?:" + cap + r" ){0,MAXL})" + re.escape(core) + r"((?: " + cap + r"){0,MAXR})(?![\\w'\\-])", p.text)
        if m and (m.group(1) or m.group(2)):
            return (m.group(1) + core + m.group(2)).strip()
    return ans'''
PATCH = [(OLD, NEW.replace("MAXL", "1").replace("MAXR", "2"))]
