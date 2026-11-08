PATCH = [('''    ans = ans.replace("*", "").strip()
''', '''    ans = ans.replace("*", "").strip()
    head, sep, _ = ans.partition(" for ")
    if sep and len(head.split()) >= 2 and not re.search(r"\\d", ans):  # "labor union for employees" -> "labor union"
        ans = head
''')]
