from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def expected_for(item: dict) -> list[str]:
    for key in ("expected_standards", "ground_truth", "relevant_standards", "standards"):
        value = item.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            return [value]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Hit Rate @3 and MRR @5 for BIS recommendations.")
    parser.add_argument("--gold", required=True, help="Public test set with ids and expected standards.")
    parser.add_argument("--pred", required=True, help="Prediction JSON from inference.py.")
    args = parser.parse_args()

    gold_items = load_json(Path(args.gold))
    if isinstance(gold_items, dict):
        gold_items = gold_items.get("queries", gold_items.get("data", [gold_items]))
    pred_items = load_json(Path(args.pred))
    pred_by_id = {str(item["id"]): item for item in pred_items}

    hit_at_3 = 0
    reciprocal_sum = 0.0
    latency_sum = 0.0

    for gold in gold_items:
        gold_id = str(gold["id"])
        expected = {normalize(item) for item in expected_for(gold)}
        pred = pred_by_id.get(gold_id, {})
        retrieved = [normalize(item) for item in pred.get("retrieved_standards", [])]
        latency_sum += float(pred.get("latency_seconds", 0.0))

        if expected.intersection(retrieved[:3]):
            hit_at_3 += 1
        for rank, standard in enumerate(retrieved[:5], start=1):
            if standard in expected:
                reciprocal_sum += 1.0 / rank
                break

    total = max(len(gold_items), 1)
    metrics = {
        "hit_rate_at_3": round(hit_at_3 / total * 100, 2),
        "mrr_at_5": round(reciprocal_sum / total, 4),
        "avg_latency_seconds": round(latency_sum / total, 6),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
