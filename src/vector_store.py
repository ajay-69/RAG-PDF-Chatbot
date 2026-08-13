import faiss
import numpy as np
import pickle
from pathlib import Path
class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[str] = []
        self.metadata: list[dict] = []

    def add(self, vectors:np.ndarray, chunks:list[str]):
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
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
        query_emb = query_emb.astype(np.float32)
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            results.append({
            "chunk": self.chunks[index],
            "score": float(score),
            "index": int(index)
            })
        return results