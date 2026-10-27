
 # format json text into a single string for embedding
@staticmethod
def context_format(context:dict):
    return " ".join([
        context["label"],
        "-",
        context["definition"],
        context["boundary"],
        " keywords: ",
        ",".join(context["keywords"]),
    ])


# compile the context database from the json file
def compile_context(context_file:dict):
    context_db = []

    for context in context_file.values():
        context_db.append(context_format(context))
    
    return context_db