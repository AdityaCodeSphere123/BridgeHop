PATCH = [('''    ans = ans.replace("*", "").strip()
''', '''    ans = ans.replace("*", "").strip()
    ans = re.sub(r"^(?:a |an |one )?(\\w+)\\s*-\\s*long$", r"\\1", ans, flags=re.I)  # "month-long" -> "month"
''')]
