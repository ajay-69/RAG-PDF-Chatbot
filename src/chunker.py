import re
class Chunker:
    def __init__(self, chunk_size:int, overlap:int):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap
    def create_chunks(self, text: str, page: int) -> list[dict]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.step):
            chunk_words = words[i : i+self.chunk_size]
            if not chunk_words:
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "page": page
                })
        return chunks

    def split(self, pages: list[dict]) -> list[dict]:
        all_chunks = []
        for page_data in pages:
            page_chunks = self.create_chunks(
                page_data["text"],
                page_data["page"]
            )
            all_chunks.extend(page_chunks)
        return all_chunks

        