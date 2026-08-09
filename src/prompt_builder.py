class PromptBuilder:

    def build(self, query: str, chunks: list[str]) -> str:

        context = "\n\n".join(
            f"[Context {i}]\n{chunk}"
            for i, chunk in enumerate(chunks, 1)
        )

        prompt = f"""
Use only the information provided in the context to answer the question.

Rules:
- Answer the question directly.
- Answer only what the question asks.
- Do not use information outside the context.
- Do not make up or assume facts.
- If the context does not contain the answer, say "I don't know."
- Keep the answer concise.

Context:
{context}
Question:
{query}
Answer:
"""

        return prompt.strip()