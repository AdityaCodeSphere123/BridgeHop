PATCH = [('''    if re.search(r"\\bcentur(y|ies)\\b", question, re.I):''', '''    long_form = re.search(r"\\b(\\w+)\\s*-\\s*long\\b", final_text)
    if long_form and re.match(r"\\s*how long\\b", question, re.I):  # "a month-long affair" -> "month"
        final_text = long_form.group(1)
    if re.search(r"\\bcentur(y|ies)\\b", question, re.I):''')]
