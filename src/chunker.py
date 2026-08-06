import re
class Chunker:
    def __init__(self, chunk_size:int, overlap:int):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap

    def split_sentences(self, text:str) -> list[str]:
        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences  

    def create_chunks(self, sentences:list[str]) -> list[str]:
        chunks = []
        for i in range(0, len(sentences), self.step):
            chunk = " ".join(sentences[i:i+self.chunk_size])
            chunks.append(chunk)
        return chunks
    def split(self, text:str) -> list[str]:
        sentences = self.split_sentences(text)
        return self.create_chunks(sentences)