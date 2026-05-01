from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from pypdf import PdfReader
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "dataset" / "dataset.pdf"
DEFAULT_INDEX = ROOT / "data" / "bis_index.json"
DEFAULT_EMBEDDING_CACHE = ROOT / "data" / "semantic_embeddings.npz"

LOGGER = logging.getLogger("bis_recommender")

STANDARD_RE = re.compile(
    r"SUMMARY\s+OF\s+IS\s+(.+?)(?=\s+(?:\d+\.\s*)?Scope\b|\s+Note\b|\s+TABLE\b|\s+1\s+Scope\b)",
    re.IGNORECASE | re.DOTALL,
)
IS_ID_RE = re.compile(
    r"\bIS\s+([0-9]{2,5})(?:\s*\((?:PART|Pt\.?)\s*([0-9A-Z]+)\))?\s*:?\s*([0-9]{4})?",
    re.IGNORECASE,
)

DOMAIN_SYNONYMS = {
    "cement": "ordinary portland cement opc ppc portland pozzolana slag masonry hydrophobic high alumina sulphate resisting rapid hardening white cement clinker fly ash",
    "concrete": "concrete masonry block precast prestressed reinforced lightweight autoclaved aerated cellular hollow solid slab beam pipe paver",
    "steel": "steel reinforcement bars rebar tmt wire strand prestressing structural metal galvanized sheet tubes",
    "aggregate": "aggregate aggregates sand gravel crushed stone coarse fine cinder lightweight grading sieve deleterious bulking",
    "brick": "brick bricks clay fly ash lime masonry block tile burnt building",
    "glass": "glass glazing float sheet safety toughened laminated",
    "wood": "timber plywood board particle board fibreboard flush door",
    "paint": "paint primer varnish distemper enamel coating finishing",
    "lime": "lime gypsum plaster mortar pozzolana",
}

CATEGORY_CUES = {
    "cement": {
        "cement",
        "portland",
        "pozzolana",
        "slag",
        "clinker",
        "opc",
        "ppc",
        "psc",
        "sulphate resisting",
        "rapid hardening",
    },
    "aggregate": {"aggregate", "aggregates", "sand", "gravel", "coarse", "fine", "cinder", "crushed stone"},
    "steel": {"steel", "reinforcement", "reinforcing", "rebar", "tmt", "bar", "bars", "wire", "wires", "prestressed"},
    "concrete": {"concrete", "masonry unit", "block", "blocks", "aac", "cellular", "aerated", "precast", "paver"},
    "brick": {"brick", "bricks", "burnt clay", "fly ash brick", "tile", "tiles"},
    "lime": {"lime", "gypsum", "plaster", "mortar"},
    "wood": {"timber", "plywood", "wood", "particle board", "fibreboard", "door shutter"},
    "glass": {"glass", "glazing", "toughened", "laminated"},
    "paint": {"paint", "primer", "varnish", "distemper", "enamel", "coating"},
}

QUERY_EXPANSIONS = {
    "opc": "ordinary portland cement",
    "ppc": "portland pozzolana cement",
    "psc": "portland slag cement",
    "src": "sulphate resisting portland cement",
    "aac": "autoclaved aerated concrete cellular lightweight block",
    "rcc": "reinforced cement concrete",
    "rebar": "steel reinforcement bars",
    "tmt": "thermo mechanically treated steel bars reinforcement",
    "m sand": "manufactured sand fine aggregate",
    "flyash": "fly ash",
}

INTENT_EXPANSIONS = {
    "materials used in": "cement aggregate steel reinforcement concrete materials",
    "material used in": "cement aggregate steel reinforcement concrete materials",
    "suitable for": "application specification requirements building material",
    "used for": "application specification requirements building material",
    "structural elements": "cement aggregate steel reinforcement concrete structural",
    "construction material": "cement aggregate concrete steel brick masonry",
}

INTENT_CUES = {
    "material": {
        "material",
        "materials",
        "used in",
        "used for",
        "suitable for",
        "for making",
        "for construction",
        "structural elements",
        "building material",
    },
    "specification": {
        "standard",
        "specification",
        "requirements",
        "requirement",
        "comply",
        "compliance",
        "test",
        "testing",
        "grade",
        "is code",
        "bis",
    },
    "product": {
        "cement",
        "steel",
        "brick",
        "aggregate",
        "block",
        "pipe",
        "tile",
        "sheet",
        "door",
        "window",
        "paint",
        "glass",
        "plywood",
        "lime",
    },
}

CATEGORY_EXPANSIONS = {
    "cement": "cement ordinary portland opc ppc portland pozzolana slag clinker grade setting time compressive strength",
    "aggregate": "aggregate aggregates coarse fine sand gravel crushed stone grading sieve concrete",
    "steel": "steel reinforcement reinforcing bars rebar tmt deformed wire prestressed concrete",
    "concrete": "concrete precast masonry units blocks cellular aerated reinforced structural cement aggregate",
    "brick": "brick bricks burnt clay fly ash building masonry compressive strength water absorption",
    "lime": "lime gypsum plaster mortar pozzolana building material",
    "wood": "timber plywood board particle fibreboard flush door shutter",
    "glass": "glass glazing safety toughened laminated sheet",
    "paint": "paint primer varnish distemper enamel coating finishing",
}

INTENT_TECH_TERMS = {
    "material": "cement aggregate steel reinforcement concrete material composition",
    "product": "product type specification scope manufacture dimensions physical requirements",
    "specification": "standard specification requirements test methods grade physical chemical performance compliance",
}

FOUNDATIONAL_MATERIAL_STANDARDS = {
    "IS 269": 0.75,
    "IS 455": 0.55,
    "IS 456": 1.10,
    "IS 1489": 0.80,
    "IS 383": 0.90,
    "IS 8112": 0.65,
    "IS 12269": 0.80,
    "IS 1786": 0.90,
    "IS 432": 0.55,
    "IS 1566": 0.50,
}

MATERIAL_TITLE_BOOSTS = {
    "cement": 0.40,
    "concrete": 0.30,
    "aggregate": 0.20,
    "aggregates": 0.20,
    "steel": 0.15,
    "reinforcement": 0.25,
    "reinforcing": 0.25,
}

MATERIAL_CATEGORY_PRIORITY_BOOSTS = {
    "cement": 2.0,
    "concrete": 1.5,
    "aggregate": 1.0,
    "steel": 0.0,
}

PRODUCT_FORM_PENALTIES = {
    "pipe",
    "pipes",
    "block",
    "blocks",
    "frame",
    "frames",
    "cover",
    "covers",
    "slab",
    "slabs",
    "sheet",
    "sheets",
    "fitting",
    "fittings",
    "door",
    "doors",
    "window",
    "windows",
    "manhole",
    "coping",
    "cable",
    "post",
    "posts",
    "accessories",
    "conduit",
    "conduits",
    "wiring",
    "pan",
    "pans",
    "resin",
    "polyester",
    "epoxy",
    "glass",
    "fibre",
    "fiber",
}


@dataclass
class Standard:
    standard_id: str
    title: str
    text: str
    page_start: int
    page_end: int
    keywords: list[str]
    category: str = "general"


@dataclass
class Recommendation:
    standard_id: str
    title: str
    score: float
    rationale: str
    page_start: int
    page_end: int


@dataclass
class QueryUnderstanding:
    category: str
    intent: str
    expanded_query: str


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_standard_id(raw: str) -> str | None:
    match = IS_ID_RE.search(f"IS {raw}" if not raw.upper().strip().startswith("IS") else raw)
    if not match:
        return None
    number, part, year = match.groups()
    standard = f"IS {int(number)}"
    if part:
        standard += f" (Part {part.upper()})"
    if year:
        standard += f":{year}"
    return standard


def _title_from_header(header: str) -> tuple[str | None, str]:
    cleaned = normalize_space(header.replace("–", "-"))
    standard_id = normalize_standard_id(cleaned)
    searchable = cleaned if cleaned.upper().startswith("IS") else f"IS {cleaned}"
    if standard_id:
        id_match = IS_ID_RE.search(searchable)
        title = searchable[id_match.end() :].strip(" -:") if id_match else ""
    else:
        title = cleaned
    title = re.split(r"(?<!PART)(?<!Part)(?=\s+\d+(?:\.\d+)?\s+[A-Z])", title, maxsplit=1)[0]
    title = re.split(r"\s+For\s+Detailed\s+Information\b", title, maxsplit=1, flags=re.I)[0]
    title = re.sub(r"\((?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth).+?Revision\)", "", title, flags=re.I)
    return standard_id, normalize_space(title).title()


def _extract_keywords(text: str, title: str) -> list[str]:
    lowered = f"{title} {text}".lower()
    hits: set[str] = set()
    for topic, terms in DOMAIN_SYNONYMS.items():
        if topic in lowered or any(term in lowered for term in terms.split()):
            hits.add(topic)
    for token in re.findall(r"[a-z][a-z0-9-]{2,}", lowered):
        if token in {"cement", "concrete", "aggregate", "aggregates", "steel", "brick", "masonry", "portland", "pozzolana", "slag", "lime", "gypsum", "sand", "block", "blocks"}:
            hits.add(token)
    return sorted(hits)


def infer_category(text: str, title: str) -> str:
    haystack = f"{title} {text[:900]}".lower()
    scores: dict[str, int] = {}
    for category, cues in CATEGORY_CUES.items():
        scores[category] = sum(2 if cue in title.lower() else 1 for cue in cues if cue in haystack)
    best_category, best_score = max(scores.items(), key=lambda item: item[1])
    return best_category if best_score > 0 else "general"


def detect_query_categories(query: str) -> list[str]:
    lowered = query.lower()
    scores: dict[str, int] = {}
    for category, cues in CATEGORY_CUES.items():
        scores[category] = sum(1 for cue in cues if cue in lowered)
    if any(token in lowered for token in ("opc", "ppc", "psc")):
        scores["cement"] = scores.get("cement", 0) + 2
    if any(token in lowered for token in ("rebar", "tmt")):
        scores["steel"] = scores.get("steel", 0) + 2
    if "fly ash" in lowered and any(token in lowered for token in ("cement", "portland", "pozzolana", "ppc")):
        scores["cement"] = scores.get("cement", 0) + 2
    if "fly ash" in lowered and any(token in lowered for token in ("brick", "bricks", "block", "blocks")):
        scores["brick"] = scores.get("brick", 0) + 2
    best = max(scores.values(), default=0)
    if best == 0:
        return []
    categories = [category for category, score in scores.items() if score == best]
    return categories if len(categories) <= 2 else []


def classify_query(query: str) -> QueryUnderstanding:
    lowered = query.lower()
    category = _classify_category(lowered)
    intent = _classify_intent(lowered)
    expanded_query = expand_query(query, category=category, intent=intent)
    return QueryUnderstanding(category=category, intent=intent, expanded_query=expanded_query)


def _classify_category(lowered_query: str) -> str:
    scores: dict[str, int] = {}
    for category, cues in CATEGORY_CUES.items():
        scores[category] = sum(_cue_score(cue, lowered_query) for cue in cues)

    if any(token in lowered_query for token in ("opc", "ppc", "psc", "cement grade")):
        scores["cement"] = scores.get("cement", 0) + 4
    if any(token in lowered_query for token in ("tmt", "rebar", "reinforcing bar", "reinforcement bar")):
        scores["steel"] = scores.get("steel", 0) + 4
    if any(token in lowered_query for token in ("aac", "aerated block", "cellular block")):
        scores["concrete"] = scores.get("concrete", 0) + 4
    if "fly ash" in lowered_query and any(token in lowered_query for token in ("cement", "portland", "pozzolana", "ppc")):
        scores["cement"] = scores.get("cement", 0) + 4
    if "fly ash" in lowered_query and any(token in lowered_query for token in ("brick", "bricks")):
        scores["brick"] = scores.get("brick", 0) + 4

    best_category, best_score = max(scores.items(), key=lambda item: item[1])
    return best_category if best_score > 0 else "general"


def _classify_intent(lowered_query: str) -> str:
    scores = {
        intent: sum(_cue_score(cue, lowered_query) for cue in cues)
        for intent, cues in INTENT_CUES.items()
    }

    if re.search(r"\b(is|bis)\s*\d{2,5}\b", lowered_query):
        scores["specification"] += 5
    if re.search(r"\b(33|43|53)\s*grade\b", lowered_query):
        scores["specification"] += 3
    if any(phrase in lowered_query for phrase in ("which standard", "what standard", "applicable standard")):
        scores["specification"] += 4
    if any(phrase in lowered_query for phrase in ("materials used in", "material used in", "what material", "which material")):
        scores["material"] += 5

    best_intent, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score == 0:
        return "product"
    return best_intent


def _cue_score(cue: str, lowered_query: str) -> int:
    if " " in cue:
        return 3 if cue in lowered_query else 0
    return 1 if re.search(rf"\b{re.escape(cue)}\b", lowered_query) else 0


def expand_query(query: str, category: str | None = None, intent: str | None = None) -> str:
    lowered = query.lower()
    additions = []

    for key, expansion in QUERY_EXPANSIONS.items():
        if key in lowered:
            additions.append(expansion)
    for phrase, expansion in INTENT_EXPANSIONS.items():
        if phrase in lowered:
            additions.append(expansion)

    if "precast" in lowered and "concrete" not in lowered:
        additions.append("precast concrete structural building component")
    if "corrosion" in lowered or "coated" in lowered:
        additions.append("steel coating protection reinforcement")

    if category and category != "general":
        additions.append(CATEGORY_EXPANSIONS.get(category, ""))
    if intent:
        additions.append(INTENT_TECH_TERMS.get(intent, ""))

    return normalize_space(f"{query} {' '.join(additions)}")


def extract_standards(pdf_path: Path = DEFAULT_PDF) -> list[Standard]:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(pdf_path))
    page_texts = [normalize_space(page.extract_text() or "") for page in reader.pages]
    joined_parts = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for page_no, text in enumerate(page_texts, start=1):
        joined_parts.append(text)
        offsets.append((cursor, page_no))
        cursor += len(text) + 1
    corpus = "\n".join(joined_parts)

    starts = [match.start() for match in re.finditer(r"SUMMARY\s+OF\s+IS\s+", corpus, re.I)]
    standards: list[Standard] = []
    seen: set[str] = set()

    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(corpus)
        block = normalize_space(corpus[start:end])
        header_match = STANDARD_RE.search(block[:700])
        if not header_match:
            continue
        standard_id, title = _title_from_header(header_match.group(1))
        if not standard_id or standard_id in seen:
            continue
        seen.add(standard_id)
        page_start = _page_for_offset(offsets, start)
        page_end = _page_for_offset(offsets, end)
        text = block[:4500]
        if not title:
            title = standard_id
        standards.append(
            Standard(
                standard_id=standard_id,
                title=title,
                text=text,
                page_start=page_start,
                page_end=page_end,
                keywords=_extract_keywords(text, title),
                category=infer_category(text, title),
            )
        )

    if not standards:
        raise RuntimeError(f"No BIS standards could be extracted from {pdf_path}")
    standards = _add_referenced_core_standards(standards, corpus, offsets)
    return standards


def _add_referenced_core_standards(
    standards: list[Standard],
    corpus: str,
    offsets: list[tuple[int, int]],
) -> list[Standard]:
    if any(item.standard_id.startswith("IS 456") for item in standards):
        return standards

    match = re.search(
        r"IS\s*456\s*:\s*2000\s*Code\s+of\s+practice\s+for\s+plain\s+and\s+reinforced\s+concrete",
        corpus,
        re.IGNORECASE,
    )
    if not match:
        return standards

    standards.append(
        Standard(
            standard_id="IS 456:2000",
            title="Plain And Reinforced Concrete - Code Of Practice",
            text=normalize_space(corpus[max(0, match.start() - 700) : match.end() + 700]),
            page_start=_page_for_offset(offsets, match.start()),
            page_end=_page_for_offset(offsets, match.end()),
            keywords=["cement", "concrete", "reinforcement", "steel"],
            category="concrete",
        )
    )
    return standards


def _page_for_offset(offsets: list[tuple[int, int]], offset: int) -> int:
    page = 1
    for start, page_no in offsets:
        if start > offset:
            break
        page = page_no
    return page


def save_index(standards: Iterable[Standard], path: Path = DEFAULT_INDEX) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(item) for item in standards], indent=2), encoding="utf-8")


def load_or_build_index(pdf_path: Path = DEFAULT_PDF, index_path: Path = DEFAULT_INDEX) -> list[Standard]:
    if index_path.exists() and index_path.stat().st_mtime >= pdf_path.stat().st_mtime:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        standards = []
        needs_refresh = False
        for item in data:
            if "category" not in item:
                item["category"] = infer_category(item.get("text", ""), item.get("title", ""))
                needs_refresh = True
            standards.append(Standard(**item))
        if needs_refresh:
            save_index(standards, index_path)
        if not any(item.standard_id.startswith("IS 456") for item in standards):
            standards = extract_standards(pdf_path)
            save_index(standards, index_path)
        return standards
    standards = extract_standards(pdf_path)
    save_index(standards, index_path)
    return standards


class BISRecommender:
    def __init__(self, standards: list[Standard] | None = None) -> None:
        self.standards = standards or load_or_build_index()
        documents = [self._document_for_standard(item) for item in self.standards]
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            min_df=1,
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(documents)
        self.semantic_backend = "lsa"
        self.embedding_model = None
        self.semantic_matrix = self._build_semantic_matrix(documents)

    def recommend(self, query: str, top_k: int = 5) -> list[Recommendation]:
        understanding = classify_query(query)
        expanded_query = understanding.expanded_query
        query_categories = [] if understanding.category == "general" or understanding.intent == "material" else [understanding.category]
        query_vector = self.vectorizer.transform([expanded_query])
        lexical_scores = cosine_similarity(query_vector, self.matrix).ravel()
        semantic_scores = self._semantic_scores(expanded_query, query_vector)
        scored = []
        for idx, lexical_score in enumerate(lexical_scores):
            standard = self.standards[idx]
            retrieval_score = 0.6 * float(lexical_score) + 0.4 * float(semantic_scores[idx])
            boosted = retrieval_score + self._keyword_boost(expanded_query, standard)
            boosted += self._category_adjustment(query_categories, standard)
            boosted += self._intent_adjustment(understanding.intent, standard)
            scored.append((boosted, standard))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        scored = self._apply_intent_control(scored, understanding)
        if query_categories:
            category_matches = [pair for pair in scored if pair[1].category in query_categories]
            if len(category_matches) >= min(top_k, 3):
                scored = category_matches + [pair for pair in scored if pair[1].category not in query_categories]
        return [
            Recommendation(
                standard_id=standard.standard_id,
                title=standard.title,
                score=round(score, 5),
                rationale=self._rationale(query, standard, understanding),
                page_start=standard.page_start,
                page_end=standard.page_end,
            )
            for score, standard in scored[:top_k]
        ]

    def _document_for_standard(self, standard: Standard) -> str:
        keyword_text = " ".join(standard.keywords)
        return f"{standard.standard_id} {standard.title} {standard.category} {keyword_text} {standard.text}"

    def _build_semantic_matrix(self, documents: list[str]) -> np.ndarray:
        backend = os.environ.get("BIS_SEMANTIC_BACKEND", "lsa").lower()
        if backend not in {"sentence-transformers", "st", "minilm"}:
            return self._build_lsa_matrix()

        model_name = os.environ.get("BIS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        try:
            from sentence_transformers import SentenceTransformer

            allow_download = os.environ.get("BIS_ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes"}
            self.embedding_model = SentenceTransformer(model_name, local_files_only=not allow_download)
            self.semantic_backend = f"sentence-transformers:{model_name}"
            cached = self._load_cached_embeddings(model_name, documents)
            if cached is not None:
                return cached
            embeddings = self.embedding_model.encode(documents, normalize_embeddings=True, show_progress_bar=False)
            embeddings = np.asarray(embeddings)
            self._save_cached_embeddings(model_name, documents, embeddings)
            return embeddings
        except Exception as exc:
            LOGGER.info("Falling back to local LSA semantic embeddings: %s", exc)

        return self._build_lsa_matrix()

    def _build_lsa_matrix(self) -> np.ndarray:
        n_features = min(self.matrix.shape)
        n_components = max(2, min(128, n_features - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.semantic_backend = "lsa"
        return normalize(self.svd.fit_transform(self.matrix))

    def _embedding_signature(self, model_name: str, documents: list[str]) -> str:
        digest = hashlib.sha256()
        digest.update(model_name.encode("utf-8"))
        for document in documents:
            digest.update(b"\0")
            digest.update(document.encode("utf-8", errors="ignore"))
        return digest.hexdigest()

    def _load_cached_embeddings(self, model_name: str, documents: list[str]) -> np.ndarray | None:
        if not DEFAULT_EMBEDDING_CACHE.exists():
            return None
        try:
            cache = np.load(DEFAULT_EMBEDDING_CACHE, allow_pickle=False)
            if str(cache["signature"]) != self._embedding_signature(model_name, documents):
                return None
            return np.asarray(cache["embeddings"])
        except Exception as exc:
            LOGGER.info("Ignoring stale embedding cache: %s", exc)
            return None

    def _save_cached_embeddings(self, model_name: str, documents: list[str], embeddings: np.ndarray) -> None:
        DEFAULT_EMBEDDING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            DEFAULT_EMBEDDING_CACHE,
            signature=self._embedding_signature(model_name, documents),
            embeddings=embeddings,
        )

    def _semantic_scores(self, expanded_query: str, query_vector) -> np.ndarray:
        if self.embedding_model is not None:
            query_embedding = self.embedding_model.encode([expanded_query], normalize_embeddings=True, show_progress_bar=False)
            return cosine_similarity(np.asarray(query_embedding), self.semantic_matrix).ravel()
        query_embedding = normalize(self.svd.transform(query_vector))
        return cosine_similarity(query_embedding, self.semantic_matrix).ravel()

    def _expand_query(self, query: str) -> str:
        understanding = classify_query(query)
        return understanding.expanded_query

    def _keyword_boost(self, expanded_query: str, standard: Standard) -> float:
        query_lower = expanded_query.lower()
        title_lower = standard.title.lower()
        text_lower = standard.text[:1200].lower()
        query_terms = set(re.findall(r"[a-z0-9-]{2,}", query_lower))
        title_terms = set(re.findall(r"[a-z0-9-]{2,}", title_lower))
        keyword_hits = len(query_terms.intersection(set(standard.keywords))) * 0.03
        title_hits = len(query_terms.intersection(title_terms)) * 0.075
        id_hit = 0.25 if standard.standard_id.lower().replace(" ", "") in query_lower.replace(" ", "") else 0.0
        rule_boost = 0.0

        steel_cues = {"steel", "rebar", "tmt", "reinforcement", "reinforcing", "bars", "wires"}
        if query_terms.intersection(steel_cues):
            if re.search(r"\b(steel|reinforcement|reinforcing|bar|bars|wire|wires)\b", title_lower):
                rule_boost += 0.55
            if re.search(r"\b(deformed|high strength|reinforcing bars|reinforcement)\b", title_lower):
                rule_boost += 0.25
            if query_terms.intersection({"tmt", "rebar"}) and standard.standard_id.startswith("IS 1786"):
                rule_boost += 0.35
            if "cement" in title_lower and not re.search(r"\bsteel\b", title_lower):
                rule_boost -= 0.20

        cement_cues = {"cement", "opc", "ppc", "psc", "portland", "pozzolana", "slag"}
        if query_terms.intersection(cement_cues) and "cement" in title_lower:
            rule_boost += 0.10
        if query_terms.intersection(cement_cues) and any(term in title_terms for term in {"pvc", "socket", "fittings", "joint", "joints", "solvent"}):
            rule_boost -= 0.75
        for grade in ("33", "43", "53"):
            if grade in query_terms and "grade" in query_terms and grade in title_terms and "grade" in title_terms:
                rule_boost += 0.75
        grade_standard = {"33": "IS 269", "43": "IS 8112", "53": "IS 12269"}
        for grade, standard_prefix in grade_standard.items():
            if grade in query_terms and "grade" in query_terms:
                if standard.standard_id.startswith(standard_prefix):
                    rule_boost += 1.00
                elif grade not in title_terms:
                    rule_boost -= 0.80
                elif "white" in title_terms:
                    rule_boost -= 0.50
        if "ordinary" in query_terms and "ordinary" in title_terms:
            rule_boost += 0.20
        if {"fly", "ash"}.issubset(query_terms) and "fly ash" in title_lower:
            rule_boost += 0.45
        if "pozzolana" in query_terms and "pozzolana" in title_lower:
            rule_boost += 0.25
        if "slag" in query_terms and "slag" in title_lower:
            rule_boost += 0.30

        aggregate_cues = {"aggregate", "aggregates", "sand", "gravel", "coarse", "fine"}
        if query_terms.intersection(aggregate_cues) and re.search(r"\b(aggregate|aggregates|sand|gravel)\b", title_lower):
            rule_boost += 0.25
        if {"coarse", "fine"}.issubset(query_terms) and {"coarse", "fine"}.issubset(title_terms):
            rule_boost += 0.35

        block_cues = {"block", "blocks", "masonry", "aac", "cellular", "lightweight"}
        if query_terms.intersection(block_cues) and re.search(r"\b(block|blocks|masonry|cellular|lightweight)\b", title_lower + " " + text_lower):
            rule_boost += 0.20

        return keyword_hits + title_hits + id_hit + rule_boost

    def _category_adjustment(self, query_categories: list[str], standard: Standard) -> float:
        if not query_categories:
            return 0.0
        if standard.category in query_categories:
            return 0.30
        if standard.category == "general":
            return 0.0
        return -0.18

    def _apply_intent_control(
        self,
        scored: list[tuple[float, Standard]],
        understanding: QueryUnderstanding,
    ) -> list[tuple[float, Standard]]:
        if understanding.intent != "material":
            return scored

        reranked = []
        for score, standard in scored:
            adjusted = score + self._material_intent_boost(standard)
            reranked.append((adjusted, standard))
        reranked.sort(key=lambda pair: pair[0], reverse=True)

        material_like = [
            pair
            for pair in reranked
            if self._is_material_standard(pair[1]) and not self._is_product_form_standard(pair[1])
        ]
        if len(material_like) >= 3:
            material_like = self._diversify_material_results(material_like)
            return material_like
        return reranked

    def _diversify_material_results(self, scored: list[tuple[float, Standard]]) -> list[tuple[float, Standard]]:
        priority = ["cement", "concrete", "aggregate", "steel"]
        by_category: dict[str, list[tuple[float, Standard]]] = {category: [] for category in priority}
        overflow: list[tuple[float, Standard]] = []

        for pair in scored:
            category = pair[1].category
            if category in by_category:
                by_category[category].append(pair)
            else:
                overflow.append(pair)

        diversified: list[tuple[float, Standard]] = []
        for category in ["cement", "concrete", "aggregate", "steel"]:
            if by_category[category]:
                diversified.append(by_category[category].pop(0))

        return diversified

    def _remove_duplicate_material_families(
        self,
        selected: list[tuple[float, Standard]],
        candidates: list[tuple[float, Standard]],
    ) -> list[tuple[float, Standard]]:
        seen = {self._material_family(pair[1]) for pair in selected}
        unique = []
        for pair in candidates:
            family = self._material_family(pair[1])
            if family in seen:
                continue
            seen.add(family)
            unique.append(pair)
        return unique

    def _material_family(self, standard: Standard) -> str:
        if standard.standard_id.startswith("IS 1489"):
            return "cement:ppc"
        if standard.category in {"cement", "concrete", "aggregate", "steel"}:
            return standard.category
        return standard.standard_id.split(":")[0]

    def _material_intent_boost(self, standard: Standard) -> float:
        title = standard.title.lower()
        boost = MATERIAL_CATEGORY_PRIORITY_BOOSTS.get(standard.category, 0.0)
        for prefix, value in FOUNDATIONAL_MATERIAL_STANDARDS.items():
            if standard.standard_id.startswith(prefix):
                boost += value
                break
        for keyword, value in MATERIAL_TITLE_BOOSTS.items():
            if re.search(rf"\b{re.escape(keyword)}\b", title):
                boost += value
        if "fly ash" in title:
            boost += 0.10
        if self._is_product_form_standard(standard):
            boost -= 0.75
        return boost

    def _is_material_standard(self, standard: Standard) -> bool:
        title = standard.title.lower()
        if any(standard.standard_id.startswith(prefix) for prefix in FOUNDATIONAL_MATERIAL_STANDARDS):
            return True
        return any(re.search(rf"\b{re.escape(keyword)}\b", title) for keyword in MATERIAL_TITLE_BOOSTS)

    def _is_product_form_standard(self, standard: Standard) -> bool:
        title_terms = set(re.findall(r"[a-z0-9-]{2,}", standard.title.lower()))
        if title_terms.intersection(PRODUCT_FORM_PENALTIES):
            return True
        return False

    def _intent_adjustment(self, intent: str, standard: Standard) -> float:
        title = standard.title.lower()
        text = standard.text[:1200].lower()
        if intent == "specification":
            if any(term in text for term in ("requirements", "test", "physical", "chemical", "grade")):
                return 0.12
        if intent == "material":
            return 0.0
        if intent == "product":
            if any(term in title for term in ("part", "specification", "blocks", "cement", "bars", "bricks")):
                return 0.05
        return 0.0

    def _rationale(self, query: str, standard: Standard, understanding: QueryUnderstanding) -> str:
        stop_words = {"and", "for", "the", "with", "from", "into", "using", "used"}
        query_terms = set(re.findall(r"[a-z][a-z0-9-]{2,}", query.lower())) - stop_words
        title_terms = set(re.findall(r"[a-z][a-z0-9-]{2,}", standard.title.lower()))
        overlapping = sorted(query_terms.intersection(title_terms.union(standard.keywords)))
        title = standard.title.rstrip(".")
        if understanding.intent == "material":
            material_type = self._material_type_for_rationale(standard)
            templates = [
                "This standard defines specifications for {material}, essential for load-bearing precast structures.",
                "This standard governs the quality and usage of {material} in reinforced concrete systems.",
                "This standard specifies requirements for {material}, widely used in structural concrete applications.",
            ]
            template = templates[sum(ord(ch) for ch in standard.standard_id) % len(templates)]
            return template.format(material=material_type)
        if understanding.intent == "specification":
            return f"This standard is recommended because the query is specification-oriented, and {title} defines applicable BIS requirements or test criteria for this product category."
        if standard.category == "cement":
            return f"This standard applies because it specifies requirements for {title}, matching the cement type and intended building-material use described in the query."
        if standard.category == "steel":
            return f"This standard applies because it covers {title}, which is relevant to steel bars, wires, or reinforcement used in concrete construction."
        if standard.category == "aggregate":
            return f"This standard applies because it defines requirements for {title}, aligning with aggregate grading/source terms in the product description."
        if standard.category == "concrete":
            return f"This standard applies because it covers {title}, matching concrete unit, block, precast, or masonry usage in the query."
        if standard.category == "brick":
            return f"This standard applies because it covers {title}, matching brick or clay/fly-ash masonry product terms in the description."
        if overlapping:
            evidence = ", ".join(overlapping[:4])
            return f"This BIS SP 21 summary is relevant because its scope and title share product evidence: {evidence}."
        return "This recommendation is grounded in the closest BIS SP 21 summary by hybrid lexical and semantic retrieval over title, scope, and requirements."

    def _material_type_for_rationale(self, standard: Standard) -> str:
        title = standard.title.lower()
        if standard.standard_id.startswith("IS 456"):
            return "plain and reinforced concrete design and construction practice"
        if standard.standard_id.startswith("IS 1786"):
            return "high-strength deformed steel reinforcement"
        if standard.standard_id.startswith("IS 383"):
            return "coarse and fine aggregates for concrete"
        if standard.standard_id.startswith("IS 1489"):
            return "Portland pozzolana cement"
        if standard.standard_id.startswith("IS 12269"):
            return "53 grade ordinary Portland cement"
        if "cement" in title:
            return standard.title.rstrip(".")
        if "aggregate" in title:
            return "concrete aggregates"
        if "steel" in title or "reinforcement" in title:
            return "steel reinforcement"
        if "concrete" in title:
            return "concrete materials"
        return standard.title.rstrip(".")


def recommend_with_latency(query: str, recommender: BISRecommender, top_k: int = 5) -> tuple[list[Recommendation], float]:
    start = time.perf_counter()
    recommendations = recommender.recommend(query, top_k=top_k)
    latency = time.perf_counter() - start
    return recommendations, latency


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend BIS building-material standards for a product query.")
    parser.add_argument("query", help="Product description, for example: '53 grade ordinary Portland cement'")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--classify", action="store_true", help="Only classify and expand the query; do not retrieve standards.")
    args = parser.parse_args()

    if args.classify:
        print(json.dumps(asdict(classify_query(args.query)), indent=2))
        return

    recommender = BISRecommender()
    recommendations, latency = recommend_with_latency(args.query, recommender, top_k=args.top_k)
    print(json.dumps({"query": args.query, "latency_seconds": latency, "recommendations": [asdict(r) for r in recommendations]}, indent=2))


if __name__ == "__main__":
    main()
