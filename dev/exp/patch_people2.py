PATCH = [('''    ans = ans.replace("*", "").strip()
''', '''    ans = ans.replace("*", "").strip()
    ans = re.sub(r"\\s+people$", "", ans, flags=re.I)  # "the Inca people" -> "the Inca"
''')]
