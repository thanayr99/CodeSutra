# BIS Standards Recommendation Engine - 8 Slide Deck

## 1. Problem Statement

Indian MSEs lose days or weeks mapping a product description to the right BIS building-material standard. The goal is to reduce that discovery step to seconds while keeping recommendations grounded in BIS SP 21.

## 2. Solution Overview

The prototype accepts a product description, retrieves the most relevant SP 21 summaries, and returns 3-5 BIS standards with short rationales. It runs fully offline with no external API dependency.

## 3. System Architecture

Product query -> lightweight category agent -> query normalization and abbreviation expansion -> hybrid retriever -> domain-aware reranker -> deterministic rationale generator -> JSON or web UI output.

## 4. Chunking & Retrieval Strategy

The PDF is chunked at `SUMMARY OF IS ...` boundaries so each retrievable unit maps to one standard. Retrieval combines TF-IDF with semantic embeddings using `0.6 * lexical + 0.4 * semantic`, then applies category filtering to reduce off-domain noise.

## 5. Demo Highlights

Example queries:

- `fly ash based portland pozzolana cement` -> `IS 1489 (Part 1):1991`
- `coarse and fine aggregates from natural sources for concrete` -> `IS 383:1970`
- `TMT rebar steel bars for concrete reinforcement` -> `IS 1786:1985`

## 6. Evaluation Results

Local sample set:

- Hit Rate @3: 100%
- MRR @5: 1.0
- Average latency after index load: about 0.018 seconds/query

## 7. Impact on MSEs

The engine gives small manufacturers a fast compliance starting point, reduces manual PDF searching, and provides traceable recommendations that can be checked against the source summary pages.

## 8. Team & Acknowledgements

Team CodeSutra. Source material: BIS SP 21 dataset supplied for the hackathon. No external APIs or non-dataset knowledge sources are used for recommendations.
