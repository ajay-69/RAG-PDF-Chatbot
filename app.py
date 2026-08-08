from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.prompt_builder import PromptBuilder
from src.generator import Generator
from src.config import EMBEDDING_MODEL, TOP_K, VECTOR_DB_PATH, LLM_NAME
device = "cuda" if torch.cuda.is_available() else "cpu"


embedding_model = EmbeddingModel(EMBEDDING_MODEL)
vector_store = VectorStore(embedding_model.dimension)
vector_store.load(VECTOR_DB_PATH)
retriever = Retriever(embedding_model, vector_store, TOP_K)
prompt_builder = PromptBuilder()
generator = Generator(LLM_NAME, device)

while True:
    query = input("ask: ")
    if query.lower() in {"exit", "quit"}:
        break
    chunks = retriever.retrieve(query)
    prompt = prompt_builder.build(query,chunks)
    ans = generator.generate(prompt)
    print("\nANSWER: ")
    print(ans)