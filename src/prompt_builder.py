class PromptBuilder:

    def build(self, question: str, retrieved_chunks: list[dict]) -> str:

        context_parts = []

        for i, result in enumerate(retrieved_chunks,1):
            context_parts.append(
                f"[Context {i}]\n"
                f"{result['chunk']}"
            ) 

        context = "\n\n".join(context_parts)
        prompt = f"""
Context:
{context}

Question:
{question}

Answer the question using only the context above.
If the answer cannot be found in the context, 
say you don't know.
"""
        return prompt