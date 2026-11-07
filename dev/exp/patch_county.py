PATCH = [('''    return {"answer": final_form(drop_unit(question, final)), "citations": citations(question, final, notes)}''',
'''    if re.search(r"\\b(what|which)\\s+(other\\s+)?county\\b", question, re.I) and final \\
            and not re.search(r"\\b(county|parish|borough|shire)\\b", final, re.I):
        final_text = f"{final} County"  # gold answers to "what county" name it as "X County"
    else:
        final_text = final
    return {"answer": final_form(drop_unit(question, final_text)), "citations": citations(question, final, notes)}''')]
