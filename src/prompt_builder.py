class PromptBuilder:

    def build(self, query: str, contexts: list[dict]) -> str:

        context_parts = []

        for i, item in enumerate(contexts, start=1):

            context_parts.append(
                f"""
[Source {i}]
Document: {item['source']}
Page: {item['page']}
Chunk ID: {item['chunk_id']}

{item['chunk']}
"""
            )

        context_text = "\n".join(context_parts)

        prompt = f"""
You are a question-answering assistant.

Answer the question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. If the answer is not present in the context, say:
   "I could not find the answer in the provided document."
3. Cite the page number for factual claims using this format:
   [Page X]
4. Do not invent page numbers or sources.
5. Keep the answer clear and concise.

Context:
{context_text}

Question:
{query}

Answer:
"""

        return prompt