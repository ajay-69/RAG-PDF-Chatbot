from src.pdf_loader import PDFLoader
from src.chunker import Chunker
from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore
from src.config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, VECTOR_DB_PATH

pdf_loader = PDFLoader()
chunker = Chunker(CHUNK_SIZE, CHUNK_OVERLAP)
embedding_model = EmbeddingModel(EMBEDDING_MODEL)
vector_store = VectorStore(embedding_model.dimension)

doc = pdf_loader.load(PDF_PATH)
chunks = chunker.split(doc)
vectors = embedding_model.encode(chunks)
vector_store.add(vectors, chunks)
vector_store.save(VECTOR_DB_PATH)