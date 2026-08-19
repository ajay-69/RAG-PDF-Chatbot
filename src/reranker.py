from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self,model_name: str, device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        pairs = []
        for result in results:
            pairs.append(
                (query, result["chunk"])
            )

        scores = self.model.predict(pairs)
        reranked = []
        for result, score in zip(results, scores):
            reranked.append({
                "chunk": result["chunk"],
                "score": float(score),
                "index": result["index"]
            })

        reranked.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return reranked[:top_k]