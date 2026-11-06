PATCH = [
    ('''    return cits[:2]''',
     '''    if not notes.answers[0] or not notes.answers[1]:  # hop 2 never gave an accepted answer: add a third passage to raise the chance
        for p in notes.ranked(1):
            if p.doc_id not in cits:
                return cits[:2] + [p.doc_id]
    return cits[:2]'''),
]
