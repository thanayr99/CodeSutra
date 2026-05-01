# BIS Standards Recommendation Engine

Proof-of-concept RAG system for the hackathon theme: turning a building-material product description into top BIS standard recommendations from BIS SP 21.

## What It Does

- Extracts `SUMMARY OF IS ...` sections from `dataset/dataset.pdf`.
- Builds a hybrid retrieval index over standard IDs, titles, scopes, and requirement summaries.
- Combines TF-IDF lexical retrieval with semantic embeddings. It uses `sentence-transformers/all-MiniLM-L6-v2` by default and caches document embeddings in `data/semantic_embeddings.npz`; if the model is unavailable, it falls back to local LSA embeddings from scikit-learn.
- Uses a lightweight category agent for cement, concrete, aggregates, steel, masonry, paint, lime, glass, and wood queries.
- Returns the top 3-5 BIS standards with deterministic rationales for demos.
- Provides the mandatory judge entry point: `python inference.py --input hidden_private_dataset.json --output team_results.json`.

No external APIs are used. The submitted PDF is the sole knowledge source.

## Repository Layout

```text
.
├── dataset/
│   └── dataset.pdf
├── src/
│   ├── __init__.py
│   └── bis_recommender.py
├── data/
│   ├── bis_index.json          # generated on first run
│   └── public_sample.json
├── app.py                      # no-dependency web demo
├── inference.py                # mandatory judge entry point
├── eval_script.py              # local metric checker
├── presentation_outline.md
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The default semantic backend is local LSA so judge runs never depend on network access. To explicitly use SentenceTransformers/MiniLM:

```bash
set BIS_SEMANTIC_BACKEND=sentence-transformers
set BIS_ALLOW_MODEL_DOWNLOAD=1
python inference.py --input data/public_sample.json --output data/sample_results.json
```

Alternative embedding model:

```bash
set BIS_SEMANTIC_BACKEND=sentence-transformers
set BIS_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
python inference.py --input data/public_sample.json --output data/sample_results.json
```

The first sentence-transformers run may download the model. After that, document embeddings are cached locally for faster startup.

## Judge Command

```bash
python inference.py --input hidden_private_dataset.json --output team_results.json
```

Input items should contain at least:

```json
[
  { "id": "q1", "query": "53 grade ordinary Portland cement for precast concrete" }
]
```

Output format:

```json
[
  {
    "id": "q1",
    "retrieved_standards": ["IS 12269:1987", "IS 8112:1989", "IS 269:1989"],
    "latency_seconds": 0.0031
  }
]
```

## Demo CLI

```bash
python -m src.bis_recommender "fly ash based portland pozzolana cement" --top-k 5
```

Classify and expand a query without retrieval:

```bash
python -m src.bis_recommender "materials used in precast structural elements" --classify
```

Output:

```json
{
  "category": "concrete",
  "intent": "material",
  "expanded_query": "materials used in precast structural elements ..."
}
```

The demo CLI includes title, score, page range, and a brief rationale. The judge output intentionally keeps only the strict fields required for automated evaluation.

## Web Demo

```bash
python app.py
```

Open `http://127.0.0.1:8000` and enter a product description. The API endpoint is also available at:

```text
http://127.0.0.1:8000/api/recommend?q=fly%20ash%20cement
```

Query understanding endpoint:

```text
http://127.0.0.1:8000/api/classify?q=materials%20used%20in%20precast%20structural%20elements
```

## Local Evaluation

```bash
python inference.py --input data/public_sample.json --output data/sample_results.json
python eval_script.py --gold data/public_sample.json --pred data/sample_results.json
```

## Retrieval Strategy

1. PDF extraction with `pypdf`.
2. Standard-level chunking: each `SUMMARY OF IS ...` block is one retrievable document.
3. Lightweight category-agent step to infer the product domain before retrieval.
4. Query expansion for common product terms such as `OPC`, `PPC`, `PSC`, `AAC`, `TMT`, and `rebar`.
5. Hybrid retrieval: `0.6 * TF-IDF lexical similarity + 0.4 * semantic embedding similarity`.
6. Category-aware reranking and noise suppression to keep cement, steel, aggregate, and masonry recommendations on-domain.

This is intentionally offline and reproducible so judges can run it on standard hardware within the latency target.
