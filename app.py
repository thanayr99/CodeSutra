from __future__ import annotations

import html
import json
import os
import time
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
      font-family: Arial, Helvetica, sans-serif;
      background: #f4f7fb;
      color: #162033;
    }
    * { box-sizing: border-box; }
    body { margin: 0; }
    main { max-width: 1120px; margin: 0 auto; padding: 28px 18px 44px; }
    header {
      display: grid;
      gap: 8px;
      padding: 12px 0 18px;
      border-bottom: 1px solid #d9e1ea;
    }
    h1 { font-size: 30px; margin: 0; letter-spacing: 0; }
    p { color: #526276; line-height: 1.5; margin: 0; }
    .query-panel {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      margin: 22px 0 18px;
      align-items: stretch;
    }
    input {
      min-height: 48px;
      border: 1px solid #b8c4d2;
      border-radius: 8px;
      padding: 0 14px;
      font-size: 16px;
      background: white;
    }
    button {
      min-height: 48px;
      border: 0;
      border-radius: 8px;
      padding: 0 18px;
      background: #1456d9;
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled { opacity: 0.65; cursor: wait; }
    .examples {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }
    .examples button {
      min-height: 34px;
      background: white;
      color: #23405f;
      border: 1px solid #c9d4e1;
      font-weight: 600;
      padding: 0 10px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 16px;
      align-items: start;
    }
    .panel, .result {
      background: white;
      border: 1px solid #d8e1eb;
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .panel { padding: 16px; }
    .panel h2 {
      font-size: 16px;
      margin: 0 0 12px;
      letter-spacing: 0;
    }
    .result { padding: 16px; margin-bottom: 12px; }
    .result-head {
      display: flex;
      gap: 10px;
      justify-content: space-between;
      align-items: flex-start;
    }
    .standard { font-weight: 800; color: #162033; line-height: 1.35; }
    .score {
      flex: 0 0 auto;
      font-size: 13px;
      color: #31506f;
      background: #eef4fb;
      border-radius: 999px;
      padding: 4px 8px;
    }
    .rationale { margin-top: 10px; color: #35465a; }
    .meta { color: #667789; font-size: 14px; margin-top: 8px; }
    .kv {
      display: grid;
      grid-template-columns: 88px 1fr;
      gap: 8px;
      font-size: 14px;
      margin-bottom: 10px;
    }
    .label { color: #667789; }
    .value { font-weight: 700; color: #162033; overflow-wrap: anywhere; }
    .expanded {
      color: #526276;
      font-size: 13px;
      line-height: 1.45;
      max-height: 180px;
      overflow: auto;
      border-top: 1px solid #e4ebf2;
      padding-top: 10px;
    }
    .empty {
      padding: 22px;
      color: #526276;
      background: white;
      border: 1px dashed #b8c4d2;
      border-radius: 8px;
    }
    @media (max-width: 820px) {
      .query-panel, .grid { grid-template-columns: 1fr; }
      h1 { font-size: 24px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>BIS Standards Recommendation Engine</h1>
      <p>Intent-aware hybrid retrieval for building-material standards. Enter a product or material description to get BIS recommendations with traceable rationale.</p>
    </header>

    <section class="query-panel">
      <input id="query" value="materials used in precast structural elements" aria-label="Product description">
      <button id="run">Recommend</button>
    </section>

    <section class="examples" aria-label="Example queries">
      <button data-query="materials used in precast structural elements">Precast materials</button>
      <button data-query="53 grade ordinary Portland cement for precast concrete">53 grade OPC</button>
      <button data-query="coarse and fine aggregates for concrete">Aggregates</button>
      <button data-query="TMT steel bars for concrete reinforcement">TMT rebar</button>
      <button data-query="fly ash bricks for masonry wall construction">Fly ash bricks</button>
    </section>

    <section class="grid">
      <div id="results" class="empty">Run a query to see the top BIS standards.</div>
      <aside class="panel">
        <h2>Query Understanding</h2>
        <div class="kv"><div class="label">Category</div><div id="category" class="value">-</div></div>
        <div class="kv"><div class="label">Intent</div><div id="intent" class="value">-</div></div>
        <div class="label">Expanded query</div>
        <div id="expanded" class="expanded">-</div>
      </aside>
    </section>
  </main>

  <script>
    const queryInput = document.querySelector("#query");
    const runButton = document.querySelector("#run");
    const resultsEl = document.querySelector("#results");
    const categoryEl = document.querySelector("#category");
    const intentEl = document.querySelector("#intent");
    const expandedEl = document.querySelector("#expanded");

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function recommend() {
      const query = queryInput.value.trim();
      if (!query) return;
      runButton.disabled = true;
      runButton.textContent = "Running";
      resultsEl.className = "empty";
      resultsEl.textContent = "Retrieving standards...";
      try {
        const response = await fetch(`/api/recommend?q=${encodeURIComponent(query)}`);
        const payload = await response.json();
        const understanding = payload.understanding || {};
        categoryEl.textContent = understanding.category || "-";
        intentEl.textContent = understanding.intent || "-";
        expandedEl.textContent = understanding.expanded_query || "-";
        renderResults(payload.recommendations || []);
      } catch (error) {
        resultsEl.className = "empty";
        resultsEl.textContent = "Unable to fetch recommendations. Check the server logs.";
      } finally {
        runButton.disabled = false;
        runButton.textContent = "Recommend";
      }
    }

    function renderResults(items) {
      if (!items.length) {
        resultsEl.className = "empty";
        resultsEl.textContent = "No recommendations found.";
        return;
      }
      resultsEl.className = "";
      resultsEl.innerHTML = items.map((item) => `
        <article class="result">
          <div class="result-head">
            <div class="standard">${escapeHtml(item.standard_id)} - ${escapeHtml(item.title)}</div>
            <div class="score">${Number(item.score).toFixed(3)}</div>
          </div>
          <div class="rationale">${escapeHtml(item.rationale)}</div>
          <div class="meta">PDF pages ${item.page_start}-${item.page_end}</div>
        </article>
      `).join("");
    }

    runButton.addEventListener("click", recommend);
    queryInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") recommend();
    });
    document.querySelectorAll("[data-query]").forEach((button) => {
      button.addEventListener("click", () => {
        queryInput.value = button.dataset.query;
        recommend();
      });
    });
    recommend();
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.write_json({"status": "ok"})
            return
        if parsed.path == "/api/recommend":
            self.handle_recommend(parsed.query)
            return
        if parsed.path == "/api/classify":
            self.handle_classify(parsed.query)
            return
        self.write_html(PAGE)

    def handle_recommend(self, raw_query: str) -> None:
        query = parse_qs(raw_query).get("q", [""])[0].strip()
        start = time.perf_counter()
        understanding = classify_query(query) if query else None
        recommendations = RECOMMENDER.recommend(query, top_k=5) if query else []
        payload = {
            "query": query,
            "understanding": asdict(understanding) if understanding else None,
            "recommendations": [asdict(rec) for rec in recommendations],
            "latency_seconds": round(time.perf_counter() - start, 6),
        }
        self.write_json(payload)

    def handle_classify(self, raw_query: str) -> None:
        query = parse_qs(raw_query).get("q", [""])[0].strip()
        payload = asdict(classify_query(query)) if query else {"category": "general", "intent": "product", "expanded_query": ""}
        self.write_json(payload)

    def write_html(self, value: str) -> None:
        body = value.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_json(self, payload: dict) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"BIS demo running at http://0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
