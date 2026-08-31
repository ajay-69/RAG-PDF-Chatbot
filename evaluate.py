import json
import torch

from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.reranker import Reranker
from src.config import (EMBEDDING_MODEL, TOP_K, RERANKER_MODEL, RERANK_TOP_K, VECTOR_DB_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Models
embedding_model = EmbeddingModel(EMBEDDING_MODEL, device)

vector_store = VectorStore(embedding_model.dimension)
vector_store.load(VECTOR_DB_PATH)
retriever = Retriever(embedding_model, vector_store, TOP_K)
reranker = Reranker(RERANKER_MODEL, device)

# Load evaluation data
with open("evaluation/evaluation.json", "r") as file:
    evaluation_data = json.load(file)

def recall_at_k(results, relevant_indices, k):
    retrieved_indices = [
        result["index"]
        for result in results[:k]
    ]
    return int(
        any(
            index in retrieved_indices
            for index in relevant_indices
        )
    )
faiss_scores = []
reranker_scores = []


for item in evaluation_data:
    question = item["question"]
    relevant = item["relevant_indices"]
    print("\n" + "=" * 80)
    print(question)
    print("=" * 80)

    # FAISS and Reranker
    results = retriever.retrieve(question)
    for result in results:
        print("Index:", result["index"], "| Page:", result["page"])
    reranked = reranker.rerank(question, results, RERANK_TOP_K)

    faiss_recall = recall_at_k(results, relevant, TOP_K)
    reranker_recall = recall_at_k(reranked, relevant, RERANK_TOP_K)

    faiss_scores.append(faiss_recall)
    reranker_scores.append(reranker_recall)

    print("Relevant:", relevant)
    print("FAISS Top", TOP_K, ":",    [x["index"] for x in results])
    print("Reranker Top", RERANK_TOP_K, ":", [x["index"] for x in reranked])
    print("FAISS Recall:", faiss_recall)
    print("Reranker Recall:",reranker_recall)

print("\n")
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(
    f"FAISS Recall@{TOP_K}: "
    f"{sum(faiss_scores) / len(faiss_scores):.2f}"
)

print(
    f"Reranker Recall@{RERANK_TOP_K}: "
    f"{sum(reranker_scores) / len(reranker_scores):.2f}"
)