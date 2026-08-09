import faiss
import numpy as np
import pickle
from pathlib import Path
class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: list[str] = []
        self.metadata: list[dict] = []

    def add(self, vectors:np.ndarray, chunks:list[str]):
        vectors = vectors.astype(np.float32)
        self.index.add(vectors)
        self.chunks.extend(chunks)
    def save(self, path:str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(Path(path)/"index.faiss"))
        with open(Path(path)/"chunks.pkl", "wb") as file:
            pickle.dump(self.chunks, file)

    def load(self, path:str) -> None:
        self.index = faiss.read_index(str(Path(path) / "index.faiss"))
        with open(Path(path) / "chunks.pkl", "rb") as file:
            self.chunks = pickle.load(file)

    def search(self, query_emb:np.ndarray, top_k:int) -> list[str]:
        distance, indices = self.index.search(query_emb, top_k)
        chunks = []
        for i in indices[0]:
            chunks.append(self.chunks[i])
        return chunks