# RAG Document Q&A Bot

A beginner-friendly Retrieval-Augmented Generation (RAG) application that answers questions from a local collection of PDF documents. The system extracts PDF text, creates overlapping chunks, generates batched embeddings, stores them persistently in ChromaDB, retrieves the most relevant chunks, and asks an LLM to generate a grounded answer with source citations.

## Architecture

PDFs → PDF extraction → overlapping chunks → batched Sentence Transformer embeddings → persistent ChromaDB → query embedding → top-k retrieval → OpenAI LLM → answer + citations.

## Tech Stack

- Python 3.11+
- Streamlit
- PyMuPDF
- Sentence Transformers
- ChromaDB
- OpenAI API
- python-dotenv

## Chunking Strategy

The project uses fixed-size character chunks of 1000 characters with 150 characters of overlap. The overlap helps preserve context when an important sentence crosses a chunk boundary.

## Embedding and Vector Database

`all-MiniLM-L6-v2` is used for local sentence embeddings. Embeddings are generated in batches instead of one chunk at a time. ChromaDB is used as the persistent vector database, stored under `vector_store/`.

## Setup

### 1. Create environment

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure OpenAI

Copy `.env.example` to `.env` and put your API key in:

```text
OPENAI_API_KEY=your_key_here
```

Never commit `.env`.

### 4. Add documents

Put 4–5 meaningful PDFs inside `data/`.

### 5. Build the index

```bash
python ingest.py
```

### 6. Start the web application

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## Example Questions

1. What is artificial intelligence?
2. What are the main types of machine learning?
3. How does deep learning work?
4. What is retrieval-augmented generation?
5. What are the applications of natural language processing?
6. Ask something that is not present in the documents.

The final question should produce the grounded fallback response instead of an unsupported answer.

## Source Citations

Each retrieved chunk stores its source filename and PDF page number. The generator is instructed to cite factual answers in the form `[filename, Page N]`.

## Requirements Covered

- PDF document ingestion
- Text extraction
- Chunking with overlap
- Chunk metadata
- Batched embeddings
- Persistent vector database
- Separate indexing and querying
- Configurable top-k retrieval
- Grounded LLM generation
- Source citations
- Interactive web UI

## Limitations

- The current loader is optimized for text-based PDFs.
- Scanned image-only PDFs require OCR.
- Answer quality depends on document quality, retrieval quality, embedding model, and the selected LLM.
- An OpenAI API key is required for answer generation.
