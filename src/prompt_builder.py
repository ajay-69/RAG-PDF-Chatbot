class PromptBuilder:

    def __init__(self):
        pass

    def build(self, query: str,chunks: list[str]) -> str:
        prompt = "you are a helpful asssistant\n"
        prompt += "Use the provided context to answer the question."
        prompt += "If the answer isn't present in the context, say you don't know.\n\n"
        prompt += "context\n"
        for i, chunk in enumerate(chunks, 1):
            prompt += f"--- Context {i} ---\n"
            prompt += chunk
            prompt += "\n"
        prompt += "query\n"
        prompt += query
        return prompt