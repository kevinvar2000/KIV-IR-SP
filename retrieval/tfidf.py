import math
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple


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

    def build(self, documents: Dict[str, str], preprocessor) -> None:
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
                term_freqs,
                self.idf,
                self.weighted_tf,
            )

    def to_dict(self) -> Dict[str, object]:
        """Serialize the index to a plain JSON-compatible dictionary."""

        return {
            "postings": self.postings,
            "doc_term_freqs": self.doc_term_freqs,
            "doc_freq": self.doc_freq,
            "idf": self.idf,
            "doc_vectors": self.doc_vectors,
            "doc_norms": self.doc_norms,
            "num_docs": self.num_docs,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "InvertedIndex":
        """Construct an index from a dictionary produced by to_dict()."""

        index = cls()
        index.postings = {
            str(term): {str(doc_id): int(tf) for doc_id, tf in docs.items()}
            for term, docs in dict(payload.get("postings", {})).items()
        }
        index.doc_term_freqs = {
            str(doc_id): {str(term): int(tf) for term, tf in terms.items()}
            for doc_id, terms in dict(payload.get("doc_term_freqs", {})).items()
        }
        index.doc_freq = {str(term): int(df) for term, df in dict(payload.get("doc_freq", {})).items()}
        index.idf = {str(term): float(idf) for term, idf in dict(payload.get("idf", {})).items()}
        index.doc_vectors = {
            str(doc_id): {str(term): float(weight) for term, weight in terms.items()}
            for doc_id, terms in dict(payload.get("doc_vectors", {})).items()
        }
        index.doc_norms = {
            str(doc_id): float(norm) for doc_id, norm in dict(payload.get("doc_norms", {})).items()
        }
        index.num_docs = int(payload.get("num_docs", 0))
        return index

    def save_json(self, file_path: str | Path) -> None:
        """Persist index structures to a JSON file."""

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(self.to_dict(), file_handle, ensure_ascii=False)

    @classmethod
    def load_json(cls, file_path: str | Path) -> "InvertedIndex":
        """Load index structures from a JSON file."""

        path = Path(file_path)
        with path.open("r", encoding="utf-8") as file_handle:
            loaded = json.load(file_handle)
        if not isinstance(loaded, dict):
            raise ValueError("Invalid index JSON format. Expected an object.")
        return cls.from_dict(loaded)

    @staticmethod
    def weighted_tf(tf: int) -> float:
        """Log-scaled TF used in the vector model."""

        if tf <= 0:
            return 0.0

        return 1.0 + math.log10(tf)


class CosineScorer:
    """Scores every document against a query using the direct cosine formula."""

    def __init__(self, index: InvertedIndex, preprocessor) -> None:
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
        """Score documents that contain query terms using inverted index and return ranked results."""

        query_tf, query_vector, query_norm = self.build_query_vector(query_text)
        if query_norm == 0:
            return []

        # Use inverted index to find candidate documents containing at least one query term.
        candidate_docs: set[str] = set()
        for term in query_tf.keys():
            if term in self.index.postings:
                candidate_docs.update(self.index.postings[term].keys())

        # Score only candidate documents (those containing at least one query term).
        scored = [
            (
                doc_id,
                cosine_similarity(
                    query_vector,
                    query_norm,
                    self.index.doc_vectors[doc_id],
                    self.index.doc_norms[doc_id],
                ),
            )
            for doc_id in candidate_docs
        ]
        scored = [(doc_id, score) for doc_id, score in scored if score > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored
