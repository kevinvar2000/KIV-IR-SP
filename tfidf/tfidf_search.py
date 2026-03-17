import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    """Return an ASCII table for pretty terminal output."""

    table_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]

    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"

    lines = [separator, header_line, separator]
    for row in table_rows:
        line = "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        lines.append(line)

    lines.append(separator)

    return "\n".join(lines)


def build_tfidf_vector(
    term_freqs: Dict[str, int],
    idf: Dict[str, float],
    weighted_tf: Callable[[int], float],
) -> Tuple[Dict[str, float], float]:
    """Build a TF-IDF vector and its L2 norm from raw term counts.

    Only keeps terms that appear in the IDF table (i.e. known index terms).
    """

    vector = {
        term: weighted_tf(tf) * idf[term]
        for term, tf in term_freqs.items()
        if term in idf
    }
    norm = math.sqrt(sum(w * w for w in vector.values()))
    return vector, norm


def cosine_similarity(
    vec_a: Dict[str, float],
    norm_a: float,
    vec_b: Dict[str, float],
    norm_b: float,
) -> float:
    """Cosine similarity between two sparse vectors: dot(a, b) / (|a| * |b|)."""

    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot = sum(vec_a.get(term, 0.0) * w for term, w in vec_b.items())
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class Collection:
    """Holds one dataset loaded from a file."""

    name: str
    documents: Dict[str, str]
    queries: Dict[str, str]


class Preprocessor:
    """Simple tokenizer used for TF-IDF without external tokenizer dependency."""

    token_pattern = re.compile(r"\b\w+\b", flags=re.UNICODE)

    def tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in self.token_pattern.findall(text)]


class CollectionParser:
    """Parses files containing lines like `d1: ...` and `q1: ...`."""

    def parse(self, file_path: str | Path) -> Collection:
        """Load one text file and split lines into documents and queries."""

        path = Path(file_path)
        documents: Dict[str, str] = {}
        queries: Dict[str, str] = {}

        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            if ":" not in stripped:
                raise ValueError(f"Invalid line in {path.name}: {line}")

            key, value = stripped.split(":", maxsplit=1)
            key = key.strip()
            value = value.strip()

            if key.lower().startswith("d"):
                documents[key] = value
            elif key.lower().startswith("q"):
                queries[key] = value
            else:
                raise ValueError(f"Unknown line prefix '{key}' in {path.name}")

        return Collection(name=path.name, documents=documents, queries=queries)


class InvertedIndex:
    """
    Stores term -> document frequencies and TF-IDF data.

    Weighting used:
    - tf(t, d): raw count in document
    - wtf(t, d): 1 + log10(tf(t, d))
    - idf(t): log10(N / df(t))
    - w(t, d): wtf(t, d) * idf(t)
    """

    def __init__(self) -> None:
        """Create empty index structures."""

        self.postings: Dict[str, Dict[str, int]] = {}
        self.doc_term_freqs: Dict[str, Dict[str, int]] = {}
        self.doc_freq: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: Dict[str, Dict[str, float]] = {}
        self.doc_norms: Dict[str, float] = {}
        self.num_docs: int = 0

    def build(self, documents: Dict[str, str], preprocessor: Preprocessor) -> None:
        """Build postings, DF/IDF, document vectors, and norms."""

        self.num_docs = len(documents)

        # First pass: compute term frequencies and build inverted index.
        for doc_id, text in documents.items():
            term_freqs: Dict[str, int] = {}

            # Tokenize and count terms for this document.
            for term in preprocessor.tokenize(text):
                term_freqs[term] = term_freqs.get(term, 0) + 1

            # Store term frequencies for later vector construction.
            self.doc_term_freqs[doc_id] = term_freqs

            # Inverted index: term -> {doc_id: tf}
            for term, tf in term_freqs.items():
                if term not in self.postings:
                    self.postings[term] = {}
                self.postings[term][doc_id] = tf

        # Second pass: compute DF, IDF, document vectors, and norms.
        self.doc_freq = {term: len(posting) for term, posting in self.postings.items()}

        # Compute IDF for each term.
        for term, df in self.doc_freq.items():
            self.idf[term] = math.log10(self.num_docs / df) if df > 0 else 0.0

        # Compute TF-IDF vectors and norms for each document.
        for doc_id, term_freqs in self.doc_term_freqs.items():
            self.doc_vectors[doc_id], self.doc_norms[doc_id] = build_tfidf_vector(
                term_freqs, self.idf, self.weighted_tf
            )

    @staticmethod
    def weighted_tf(tf: int) -> float:
        """Log-scaled TF used in the vector model."""

        if tf <= 0:
            return 0.0

        return 1.0 + math.log10(tf)


class CosineScorer:
    """Scores every document against a query using the direct cosine formula."""

    def __init__(self, index: InvertedIndex, preprocessor: Preprocessor) -> None:
        self.index = index
        self.preprocessor = preprocessor

    def build_query_vector(self, query_text: str) -> Tuple[Dict[str, int], Dict[str, float], float]:
        """Tokenize the query and build its TF-IDF vector and norm."""

        tf: Dict[str, int] = {}
        for term in self.preprocessor.tokenize(query_text):
            tf[term] = tf.get(term, 0) + 1

        vector, norm = build_tfidf_vector(tf, self.index.idf, self.index.weighted_tf)
        return tf, vector, norm

    def search(self, query_text: str) -> List[Tuple[str, float]]:
        """Score every document with the direct cosine formula and return ranked results."""

        _, query_vector, query_norm = self.build_query_vector(query_text)
        if query_norm == 0:
            return []

        # Compute cosine similarity directly for every document (manual-style formula).
        scored = [
            (doc_id, cosine_similarity(query_vector, query_norm, doc_vector, self.index.doc_norms[doc_id]))
            for doc_id, doc_vector in self.index.doc_vectors.items()
        ]
        scored = [(doc_id, score) for doc_id, score in scored if score > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored


def build_term_breakdown_rows(
    term_freqs: Dict[str, int],
    index: InvertedIndex,
    vector: Dict[str, float],
    norm: float,
) -> List[List[str]]:
    """Create per-term rows used by the printed TF-IDF tables."""

    rows: List[List[str]] = []
    for term in sorted(term_freqs.keys()):

        tf = term_freqs[term]
        wtf = index.weighted_tf(tf)
        tf_idf = vector.get(term, 0.0)

        normalized = tf_idf / norm if norm != 0 else 0.0
        rows.append([
            term,
            str(tf),
            f"{wtf:.6f}",
            f"{tf_idf:.6f}",
            f"{normalized:.6f}",
        ])

    return rows


def run_collection(file_path: str | Path) -> None:
    """Run the full pipeline for one file and print all reports."""

    parser = CollectionParser()
    preprocessor = Preprocessor()

    collection = parser.parse(file_path)
    index = InvertedIndex()
    index.build(collection.documents, preprocessor)

    scorer = CosineScorer(index, preprocessor)

    print(f"\n=== {collection.name} ===")
    print("\nTerm DF / IDF")
    term_rows = [
        [term, str(index.doc_freq[term]), f"{index.idf[term]:.6f}"]
        for term in sorted(index.postings.keys())
    ]
    print(render_table(["term", "DF", "IDF"], term_rows))

    print("\nDocument vectors")
    for doc_id in sorted(collection.documents.keys()):

        # Per-document vector details for manual inspection.
        print(f"\n{doc_id}: {collection.documents[doc_id]}")
        rows = build_term_breakdown_rows(
            index.doc_term_freqs[doc_id],
            index,
            index.doc_vectors[doc_id],
            index.doc_norms[doc_id],
        )
        print(render_table(["term", "TF", "weighted TF", "TF-IDF", "normed"], rows))
        print(f"||{doc_id}|| = {index.doc_norms[doc_id]:.6f}")

    for query_id, query_text in collection.queries.items():

        # Query vector and ranking against document vectors.
        query_tf, query_vector, query_norm = scorer.build_query_vector(query_text)
        ranked = scorer.search(query_text)

        print(f"\n{query_id}: {query_text}")
        q_rows = build_term_breakdown_rows(query_tf, index, query_vector, query_norm)
        print(render_table(["term", "TF", "weighted TF", "TF-IDF", "normed"], q_rows))
        print(f"||{query_id}|| = {query_norm:.6f}")

        print("Ranking")
        if not ranked:
            print("  No matching documents.")
            print("Best relevant document: N/A")

        for doc_id, score in ranked:
            print(f"  {doc_id}: {score:.6f}")

        if ranked:
            best_doc, best_score = ranked[0]
            print(f"Best relevant document: {best_doc} (score={best_score:.6f})")


if __name__ == "__main__":
    run_collection("test1.txt")
    run_collection("test2.txt")
