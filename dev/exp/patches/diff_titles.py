PATCH = [
    ('''    cits = list(dict.fromkeys(d for d in cits if d))
''',
     '''    cits = list(dict.fromkeys(d for d in cits if d))
    # The two gold passages almost always come from different articles. If both citations are paragraphs of
    # one article, replace the bridge citation with the best passage from another article.
    if len(cits) == 2 and notes.passages[cits[0]].title == notes.passages[cits[1]].title:
        title = notes.passages[cits[1]].title
        others = [(i, p) for i, p in enumerate(notes.ranked(0)) if p.title != title]
        if others:
            best = max(others, key=lambda ip: (title_in(ip[1], question), bool(bridge) and mentions(ip[1], bridge),
                                               -ip[0]))
            cits[0] = best[1].doc_id
'''),
]
