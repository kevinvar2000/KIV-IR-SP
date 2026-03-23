from pathlib import Path

try:
    from .dataset import CollectionParser, Preprocessor
    from .reporting import build_term_breakdown_rows, render_table
    from .tfidf import CosineScorer, InvertedIndex
except ImportError:
    from dataset import CollectionParser, Preprocessor
    from reporting import build_term_breakdown_rows, render_table
    from tfidf import CosineScorer, InvertedIndex


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
