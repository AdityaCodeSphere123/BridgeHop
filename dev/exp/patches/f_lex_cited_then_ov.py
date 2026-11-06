OLD_SD = 'def supporting_doc(ans: str, cited, candidates: list[Passage], avoid=None, prefer=None):'
NEW_SD = '''def overlap(p: Passage, words: set[str]) -> int:
    """How many of the words appear in the passage (title or text)."""
    return len(words & set(norm(p.title + " " + p.text).split()))


def supporting_doc(ans: str, cited, candidates: list[Passage], avoid=None, prefer=None, words=frozenset()):'''
W = 1.0
PATCH = [
    (OLD_SD, NEW_SD),
    ('        return (W_PREFER * bool(prefer and prefer(p)) + W_CITED * is_cited, -rank)',
     '        return (bool(prefer and prefer(p)), is_cited, overlap(p, words), -rank)'),
    ('                                   prefer=lambda p: title_in(p, question)))',
     '                                   prefer=lambda p: title_in(p, question), words=content_words(notes.subqs[0])))'),
    ('                                   prefer=lambda p: bool(bridge) and mentions(p, bridge)))',
     '                                   prefer=lambda p: bool(bridge) and mentions(p, bridge),\n'
     '                                   words=content_words(notes.subqs[1].replace("#1", "")) | content_words(question)))'),
]
