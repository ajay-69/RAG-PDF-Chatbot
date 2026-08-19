from src.pdf_loader import PDFLoader
from src.chunker import Chunker
from src.config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, VECTOR_DB_PATH
from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

pdf_loader = PDFLoader()
chunker = Chunker(CHUNK_SIZE, CHUNK_OVERLAP)
embedding_model = EmbeddingModel(EMBEDDING_MODEL, device)
vector_store = VectorStore(embedding_model.dimension)

doc = pdf_loader.load(PDF_PATH)

print("Characters:", len(doc))

chunks = chunker.split(doc)

print("Total chunks:", len(chunks))

vectors = embedding_model.encode(chunks)

print("Embedding shape:", vectors.shape)

vector_store.add(
    vectors,
    chunks
)

print(
    "Total vectors:",
    vector_store.index.ntotal
)

vector_store.save(VECTOR_DB_PATH)

print("Vector store saved.")