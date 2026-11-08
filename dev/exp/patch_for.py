PATCH = [('''    if re.search(r"\\bcentur(y|ies)\\b", question, re.I):''', '''    head, sep, _ = final_text.partition(" for ")
    if sep and len(head.split()) >= 2 and not re.search(r"\\d", final_text):  # "labor union for employees" -> "labor union"
        final_text = head
    if re.search(r"\\bcentur(y|ies)\\b", question, re.I):''')]
