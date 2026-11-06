PATCH = [('''    return len(words & set(norm(p.title + " " + p.text).split()))''', '''    return len(words & set(norm(p.title + " " + p.text).split())) + len(words & set(norm(p.title).split()))''')]
