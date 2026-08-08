class Retriever:
    def __init__(self, embedding_model, vector_store, top_k:int):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query:str) -> list[str]:
        query_emb = self.embedding_model.encode([query])
        return self.vector_store.search(query_emb, self.top_k)