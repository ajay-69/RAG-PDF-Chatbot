import torch
from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.prompt_builder import PromptBuilder
from src.generator import Generator
from src.config import EMBEDDING_MODEL, TOP_K, RERANKER_MODEL, RERANK_TOP_K, VECTOR_DB_PATH, LLM_NAME
from src.reranker import Reranker
device = "cuda" if torch.cuda.is_available() else "cpu"


embedding_model = EmbeddingModel(EMBEDDING_MODEL, device)
vector_store = VectorStore(embedding_model.dimension)
vector_store.load(VECTOR_DB_PATH)
retriever = Retriever(embedding_model, vector_store, TOP_K)
reranker = Reranker(RERANKER_MODEL, device)
prompt_builder = PromptBuilder()
generator = Generator(LLM_NAME, device)

while True:
    query = input("\nask: ")
    if query.lower() in {"exit", "quit"}:
        break

    chunks = retriever.retrieve(query)
    print("\n========== FAISS RESULTS ==========")
    for i, result in enumerate(chunks, 1):
        print(f"\n--- FAISS Rank {i} ---")
        print("Index:", result["index"])
        print("Page:", result["page"])
        print("Distance:", result["score"])
        print("Text:", result["chunk"][:500])


    reranked_chunks = reranker.rerank(query, chunks, RERANK_TOP_K)
    print("\n========== RERANKED RESULTS ==========")
    for i, result in enumerate(reranked_chunks, 1):
        print(f"\n--- Reranked Rank {i} ---")
        print("Index:", result["index"])
        print("Page:", result["page"])
        print("Score:", result["score"])
        print("Text:", result["chunk"][:500])

    prompt = prompt_builder.build(query,reranked_chunks)
    ans = generator.generate(prompt)
    print("\nANSWER:")
    print(ans)