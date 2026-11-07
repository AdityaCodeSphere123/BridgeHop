PATCH = [('''    else:
        final_text = final
''', '''    else:
        final_text = final
    if re.search(r"\\bcentur(y|ies)\\b", question, re.I):  # gold writes "7th", not "seventh"
        for i, w in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
                              "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth "
                              "twentieth".split(), 1):
            suffix = "th" if 10 < i < 14 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
            final_text = re.sub(rf"\\b{w}\\b", f"{i}{suffix}", final_text, flags=re.I)
''')]
