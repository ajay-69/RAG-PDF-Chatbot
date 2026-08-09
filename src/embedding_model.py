from sentence_transformers import SentenceTransformer
import numpy as np
class EmbeddingModel:
    def __init__(self, model_name:str):
        self.model = SentenceTransformer(model_name)

    def encode(self, chunks:list[str]) -> np.ndarray:
        return self.model.encode(chunks, batch_size=32, show_progress_bar=True)
    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()