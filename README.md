# PDF RAG Chatbot

A local, command-line Retrieval-Augmented Generation (RAG) chatbot for asking questions about a PDF. It extracts the document text, stores semantic embeddings in FAISS, reranks retrieved passages, and generates grounded answers from the supplied context.

The project is configured with `data/RAG_pdf.pdf`, a document about the EU Artificial Intelligence Act.

## Pipeline

```text
PDF -> PyMuPDF text extraction -> overlapping chunks -> BGE embeddings -> FAISS
                                                                         |
Question -> BGE embedding -> top-10 retrieval -> cross-encoder reranking -> top-5 context -> Qwen answer
```

## Features

- Page-aware PDF text extraction with PyMuPDF.
- Word-based chunks of 300 words with 50-word overlap.
- Semantic retrieval with `BAAI/bge-small-en-v1.5` and FAISS L2 search.
- Cross-encoder reranking with `BAAI/bge-reranker-base`.
- Context-grounded generation using `Qwen/Qwen2.5-1.5B-Instruct`.
- CUDA is used automatically when it is available; otherwise the app runs on CPU.
- A small evaluation set reports retrieval and reranking Recall@K.

## Project structure

```text
.
|-- app.py                    # Interactive question-answering CLI
|-- ingest.py                 # Builds the FAISS index from the PDF
|-- evaluate.py               # Measures retrieval and reranking Recall@K
|-- requirements.txt
|-- data/
|   `-- RAG_pdf.pdf           # Source document
|-- evaluation/
|   `-- evaluation.json       # Questions and relevant chunk indices
|-- vector_db/                # Generated FAISS index and stored metadata
`-- src/
    |-- config.py             # Paths, model names, and retrieval settings
    |-- pdf_loader.py         # PDF extraction
    |-- chunker.py            # Overlapping word chunking
    |-- embedding_model.py    # SentenceTransformer wrapper
    |-- vector_store.py       # FAISS persistence and search
    |-- retriever.py
    |-- reranker.py
    |-- prompt_builder.py
    `-- generator.py
```

## Setup

Use Python 3.10 or newer and create a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On the first run, Hugging Face downloads the embedding, reranker, and generation models. Ensure you have sufficient disk space and an internet connection. A CUDA-capable PyTorch installation is optional; the code falls back to CPU automatically.

## Build the vector database

Run ingestion whenever the source PDF or chunking configuration changes:

```powershell
python ingest.py
```

This creates or replaces the persisted index files in `vector_db/`:

- `index.faiss`
- `chunks.pkl`
- `metadata.pkl`

## Ask questions

After ingestion completes, start the chatbot:

```powershell
python app.py
```

Example:

```text
ask: What is a high-risk AI system?
```

Type `exit` or `quit` to close the application.

## Evaluate retrieval

```powershell
python evaluate.py
```

The script evaluates five sample questions in `evaluation/evaluation.json`. It prints the chunks selected by FAISS and by the reranker, then reports a hit-based Recall@10 for FAISS and Recall@5 after reranking. Relevant chunk indices must be updated if you regenerate the index with a changed PDF or chunking configuration.

## Configuration

Edit [`src/config.py`](src/config.py) to change the source PDF, chunking settings, models, database path, or number of passages retrieved and reranked.

```python
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 10
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_TOP_K = 5
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
```

## Notes

- Answers are instructed to use only retrieved context. If the answer is absent, the model is asked to say so.
- The PDF loader preserves page metadata internally, which is available in retrieval results and useful for debugging or extending the UI with citations.
- `vector_db/` is derived data. Re-run `python ingest.py` rather than manually editing its files.
