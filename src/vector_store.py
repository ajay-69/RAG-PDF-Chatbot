import faiss
import numpy as np
import pickle
from pathlib import Path
class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: list[str] = []
        self.metadata: list[dict] = []

    def add(self, vectors:np.ndarray, chunks:list[dict]) -> None:
        vectors = vectors.astype(np.float32)
        self.index.add(vectors)
        self.chunks.extend(
            [chunks["text"] for chunk in chunks]
        )
        self.metadata.extend(
            [
                {"page": chunk["page"]}
                for chunk in chunks
            ]
        )

    def save(self, path:str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path/"index.faiss"))
        with open(path/"chunks.pkl", "wb") as file:
            pickle.dump(self.metadata, file)

    def load(self, path:str) -> None:
        path = Path(path)
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as file:
            self.chunks = pickle.load(file)
        with open(path/"metadata.pkl", "rb") as file:
            self.metadata = pickle.load(file)

    def search(self, query_emb:np.ndarray, top_k:int) -> list[str]:
        query_emb = query_emb.astype(np.float32)
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for score, index in zip(distances[0], indices[0]):
            if index == -1:
                continue
            results.append({
            "chunk": self.chunks[index],
            "score": float(score),
            "index": int(index),
             "page": self.metadata[index]["page"]
            })
        return results