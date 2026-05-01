from __future__ import annotations

import html
import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from src.bis_recommender import BISRecommender, classify_query


RECOMMENDER = BISRecommender()


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BIS Standards Recommendation Engine</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, sans-serif;
      background: #f6f7f9;
      color: #17202a;
    }
    body { margin: 0; }
    main { max-width: 980px; margin: 0 auto; padding: 32px 20px 48px; }
    h1 { font-size: 30px; margin: 0 0 8px; letter-spacing: 0; }
    p { color: #516070; line-height: 1.5; }
    form { display: grid; grid-template-columns: 1fr auto; gap: 12px; margin: 24px 0; }
    input {
      min-height: 44px;
      border: 1px solid #bcc6d0;
      border-radius: 6px;
      padding: 0 14px;
      font-size: 16px;
      background: white;
    }
    button {
      min-height: 44px;
      border: 0;
      border-radius: 6px;
      padding: 0 18px;
      background: #1f6feb;
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    .result {
      background: white;
      border: 1px solid #d8dee6;
      border-radius: 8px;
      padding: 16px;
      margin: 12px 0;
    }
    .standard { font-weight: 700; }
    .meta { color: #667789; font-size: 14px; margin-top: 6px; }
    @media (max-width: 640px) {
      form { grid-template-columns: 1fr; }
      h1 { font-size: 24px; }
    }
  </style>
</head>
<body>
  <main>
    <h1>BIS Standards Recommendation Engine</h1>
    <p>Enter a building-material product description to retrieve relevant BIS SP 21 standards with short evidence-based rationales.</p>
    <form method="get">
      <input name="q" value="__QUERY__" placeholder="Example: fly ash based portland pozzolana cement" autofocus>
      <button type="submit">Recommend</button>
    </form>
    __RESULTS__
  </main>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/recommend":
            self.handle_api(parsed.query)
            return
        if parsed.path == "/api/classify":
            self.handle_classify(parsed.query)
            return
        query = parse_qs(parsed.query).get("q", [""])[0].strip()
        results = self.render_results(query) if query else ""
        body = PAGE.replace("__QUERY__", html.escape(query)).replace("__RESULTS__", results).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_api(self, raw_query: str) -> None:
        query = parse_qs(raw_query).get("q", [""])[0].strip()
        understanding = classify_query(query) if query else None
        recommendations = RECOMMENDER.recommend(query, top_k=5) if query else []
        payload = {
            "query": query,
            "understanding": asdict(understanding) if understanding else None,
            "recommendations": [rec.__dict__ for rec in recommendations],
        }
        self.write_json(payload)

    def handle_classify(self, raw_query: str) -> None:
        query = parse_qs(raw_query).get("q", [""])[0].strip()
        payload = asdict(classify_query(query)) if query else {"category": "general", "intent": "product", "expanded_query": ""}
        self.write_json(payload)

    def write_json(self, payload: dict) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def render_results(self, query: str) -> str:
        cards = []
        for rec in RECOMMENDER.recommend(query, top_k=5):
            cards.append(
                f"""
                <section class="result">
                  <div class="standard">{html.escape(rec.standard_id)} - {html.escape(rec.title)}</div>
                  <p>{html.escape(rec.rationale)}</p>
                  <div class="meta">Score {rec.score:.3f} | PDF pages {rec.page_start}-{rec.page_end}</div>
                </section>
                """
            )
        return "\n".join(cards)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("BIS demo running at http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
