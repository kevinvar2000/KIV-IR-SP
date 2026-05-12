"""Core TF-IDF indexing, vectorization, and cosine scoring primitives."""

import math
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple


from indexing.inverted_index import InvertedIndex, build_tfidf_vector

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


class CosineScorer:
    """Scores every document against a query using the direct cosine formula."""

    def __init__(self, index: InvertedIndex, preprocessor) -> None:
        """Bind scorer to an index and query tokenizer/preprocessor."""
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


class BM25Scorer:
    """Scores every document against a query using the Okapi BM25 algorithm."""

    def __init__(self, index: InvertedIndex, preprocessor, k1: float = 1.5, b: float = 0.75) -> None:
        """Bind scorer to an index and query tokenizer/preprocessor."""
        self.index = index
        self.preprocessor = preprocessor
        self.k1 = k1
        self.b = b

    def search(self, query_text: str) -> List[Tuple[str, float]]:
        """Score documents using BM25 and return ranked results."""
        query_tf: Dict[str, int] = {}
        for term in self.preprocessor.tokenize(query_text):
            query_tf[term] = query_tf.get(term, 0) + 1

        if not query_tf:
            return []

        scores: Dict[str, float] = {}
        N = self.index.num_docs
        avgdl = self.index.avgdl
        if avgdl == 0:
            avgdl = 1.0

        for term, q_tf in query_tf.items():
            if term not in self.index.postings:
                continue
            
            df = self.index.doc_freq.get(term, 0)
            # Standard BM25 IDF formulation
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
            if idf < 0:
                idf = 0.0
            
            for doc_id, tf in self.index.postings[term].items():
                dl = self.index.doc_lengths.get(doc_id, avgdl)
                if dl == 0:
                    dl = 1.0

                tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (dl / avgdl)))
                scores[doc_id] = scores.get(doc_id, 0.0) + (idf * tf_component * q_tf)

        scored = [(doc_id, score) for doc_id, score in scores.items() if score > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored
