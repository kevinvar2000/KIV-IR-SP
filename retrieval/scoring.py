"""Backward-compatible TF-IDF exports.

Use retrieval.tfidf directly for new code.
"""

try:
    from .tfidf import CosineScorer, InvertedIndex, build_tfidf_vector, cosine_similarity
except ImportError:
    from tfidf import CosineScorer, InvertedIndex, build_tfidf_vector, cosine_similarity

__all__ = [
    "build_tfidf_vector",
    "cosine_similarity",
    "InvertedIndex",
    "CosineScorer",
]
