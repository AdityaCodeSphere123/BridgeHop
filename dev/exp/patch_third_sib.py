PATCH = [('''    return cits[:2]''', '''    cits = cits[:2]
    if len(cits) == 2 and notes.passages[cits[0]].title == notes.passages[cits[1]].title:
        # Both are paragraphs of one article, but the gold evidence spans two: add the best other article.
        title = notes.passages[cits[0]].title
        other = next((p for p in notes.ranked(1) if p.title != title and (not notes.answers[0]
                      or mentions(p, notes.answers[0]) or mentions(p, final))), None)
        if other is not None:
            cits.append(other.doc_id)
    return cits''')]
