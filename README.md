# BIS Standards Recommendation Engine

Intent-aware hybrid retrieval system for the BIS x Sigma Squad AI Hackathon. It turns building-material product descriptions into relevant BIS standard recommendations from the supplied BIS SP 21 PDF.

## What It Does

- Extracts `SUMMARY OF IS ...` sections from `dataset/dataset.pdf`.
- Builds a hybrid retrieval index over standard IDs, titles, scopes, and requirement summaries.
- Uses TF-IDF plus local semantic LSA embeddings by default, so judge runs do not need network access.
- Classifies category and intent before retrieval, for example `cement`, `steel`, `aggregate`, `concrete`, `material`, `product`, or `specification`.
- Applies intent-aware reranking so material queries return foundational materials such as cement, concrete, aggregates, and reinforcement instead of only product standards.
- Provides the mandatory judge entry point: `python inference.py --input hidden_private_dataset.json --output team_results.json`.
- Serves a real-time frontend and backend from `app.py`.

No external APIs are used by default. The submitted PDF is the sole recommendation source.

## Repository Layout

```text
dataset/
  dataset.pdf
src/
  __init__.py
  bis_recommender.py
data/
  bis_index.json
  public_sample.json
  sample_results.json
app.py
inference.py
eval_script.py
presentation_outline.md
requirements.txt
requirements-embeddings.txt
render.yaml
Procfile
runtime.txt
README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional SentenceTransformers backend:

```bash
pip install -r requirements-embeddings.txt
set BIS_SEMANTIC_BACKEND=sentence-transformers
set BIS_ALLOW_MODEL_DOWNLOAD=1
```

## Judge Command

```bash
python inference.py --input hidden_private_dataset.json --output team_results.json
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

## Local Evaluation

```bash
python inference.py --input data/public_sample.json --output data/sample_results.json
python eval_script.py --gold data/public_sample.json --pred data/sample_results.json
```

## CLI Demo

Recommend standards:

```bash
python -m src.bis_recommender "materials used in precast structural elements" --top-k 5
```

Classify and expand a query:

```bash
python -m src.bis_recommender "materials used in precast structural elements" --classify
```

Example classification:

```json
{
  "category": "concrete",
  "intent": "material",
  "expanded_query": "materials used in precast structural elements ..."
}
```

## Web Demo

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8000
```

The UI shows:

- query category
- query intent
- expanded query
- ranked BIS standards
- PDF page references
- human-readable rationale

API endpoints:

```text
GET /api/recommend?q=fly%20ash%20cement
GET /api/classify?q=materials%20used%20in%20precast%20structural%20elements
GET /health
```

## Hosting

The app is deployable as a single Python web service. `app.py` serves both the frontend and backend.

Render setup:

```text
Build command: pip install -r requirements.txt
Start command: python app.py
Health check path: /health
```

The repository includes `render.yaml`, `Procfile`, and `runtime.txt`. The server reads the hosting provider's `PORT` environment variable automatically.

## Retrieval Strategy

1. PDF extraction with `pypdf`.
2. Standard-level chunking at each `SUMMARY OF IS ...` section.
3. Query understanding: category plus intent.
4. Query expansion for domain abbreviations and material intent.
5. Hybrid retrieval: `0.6 * TF-IDF similarity + 0.4 * semantic similarity`.
6. Intent-aware filtering, material-family diversity, and domain reranking.

Demo line:

> Our system understands not just keywords but intent. For material-based queries, it intelligently shifts retrieval towards foundational construction components like cement, aggregates, concrete, and reinforcement, ensuring domain-accurate BIS recommendations.
