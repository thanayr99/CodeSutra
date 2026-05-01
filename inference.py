from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.bis_recommender import BISRecommender, recommend_with_latency


def load_queries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "queries" in data:
            return list(data["queries"])
        if "data" in data:
            return list(data["data"])
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Input JSON must be a list of query objects or an object containing 'queries'.")


def item_query(item: dict) -> str:
    for key in ("query", "product_description", "description", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"Input item {item.get('id', '<missing id>')} does not contain a query field.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge entry point for BIS recommendations.")
    parser.add_argument("--input", required=True, help="Path to hidden/private dataset JSON.")
    parser.add_argument("--output", required=True, help="Path to write team_results.json.")
    args = parser.parse_args()

    queries = load_queries(Path(args.input))
    recommender = BISRecommender()
    results = []

    for index, item in enumerate(queries):
        query = item_query(item)
        recommendations, latency = recommend_with_latency(query, recommender, top_k=5)
        results.append(
            {
                "id": item.get("id", str(index)),
                "retrieved_standards": [rec.standard_id for rec in recommendations],
                "latency_seconds": round(latency, 6),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
