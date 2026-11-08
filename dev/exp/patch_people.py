PATCH = [('''    if re.search(r"\\bcentur(y|ies)\\b", question, re.I):''', '''    final_text = re.sub(r"\\s+people$", "", final_text, flags=re.I)  # "the Inca people" -> "the Inca"
    if re.search(r"\\bcentur(y|ies)\\b", question, re.I):''')]
