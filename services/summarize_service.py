def generate_summary(llm, content, summary_type="general"):
    prompt = """
    Summarize the following content in clear and simple language:
    {context}
    """
    if summary_type == "key-points":
        prompt = """
        List the key points from the following content:
        {context}
        """
    elif summary_type == "detailed":
        prompt = """
        Provide a detailed summary of the following content:
        {context}
        """
    elif summary_type == "bullet-points":
        prompt = """
        Summarize the following content as bullet points:
        {context}
        """
    final_prompt = prompt.format(context=content)
    return llm.invoke(final_prompt)