"""Interactive query interface for TF-IDF and Boolean search."""

import json
import re
from pathlib import Path

from preprocessing.config import build_pipelines
from preprocessing.tokenizer import RegexMatchTokenizer

from .tfidf import CosineScorer, InvertedIndex
from .boolean import BooleanIndex, BooleanScorer
from .dataset import Preprocessor


class PipelineQueryPreprocessor:
    """Apply the same preprocessing pipeline to queries as used for index generation."""

    def __init__(self, pipeline_name: str):
        pipelines = build_pipelines()
        if pipeline_name not in pipelines:
            raise ValueError(f"Unknown preprocessing pipeline for query mode: {pipeline_name}")
        self.pipeline = pipelines[pipeline_name]
        self.tokenizer = RegexMatchTokenizer()

    def tokenize(self, text: str) -> list[str]:
        tokens = self.tokenizer.tokenize(text)
        tokens = self.pipeline.preprocess(tokens, text)
        return [token.processed_form for token in tokens if token.processed_form]


def _format_results(results: list[tuple[str, float]], method: str, metadata: dict[str, dict] | None = None, max_display: int = 10) -> str:
    """Format search results nicely with rank | score | title | link."""
    if not results:
        return "No matching documents found."

    lines = []
    limited = results[:max_display]
    
    # Header
    if method == "tfidf":
        lines.append("=" * 60)
        lines.append("│ Rank │ Score │ Title                             │ Link")
        lines.append("-" * 60)
    else:
        lines.append("=" * 60)
        lines.append("│ Rank   │ Title                             │ Link")
        lines.append("-" * 60)
    
    for rank, (doc_id, score) in enumerate(limited, start=1):
        doc_url = doc_id if doc_id.startswith("http") else doc_id
        
        # Get metadata if available
        title = "Unknown"
        if metadata and doc_url in metadata:
            title = metadata[doc_url].get("title", "Unknown")[:30]
        
        # Truncate title if too long
        if len(title) > 30:
            title = title[:27] + "..."
        
        if method == "tfidf":
            score_str = f"{score:.4f}".rjust(6)
            lines.append(f"│ {rank:2d}  │ {score_str} │ {title:33s} │ {doc_url[:30]:30s}")
        else:  # boolean
            lines.append(f"│ {rank:2d}  │ {title:33s} │ {doc_url[:30]:30s}")

    # Footer
    lines.append("=" * 60)
    
    if len(results) > max_display:
        lines.append(f"\n  ... and {len(results) - max_display} more results")
    
    return "\n".join(lines)


def _extract_query_terms(query: str, query_preprocessor: PipelineQueryPreprocessor | None = None) -> list[str]:
    """Extract terms to highlight in document debug snippets."""
    if query_preprocessor is not None:
        try:
            tokens = query_preprocessor.tokenize(query)
            if tokens:
                return list(dict.fromkeys(tokens))
        except Exception:
            pass

    terms: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9_]+", query):
        lowered = raw.lower()
        if lowered in {"and", "or", "not"}:
            continue
        terms.append(lowered)
    return list(dict.fromkeys(terms))


def _highlight_text(text: str, terms: list[str]) -> str:
    """Highlight matched terms in text using [[term]] markers."""
    if not text or not terms:
        return text

    highlighted = text
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"[[{m.group(0)}]]", highlighted)
    return highlighted


def _build_debug_snippet(doc_id: str, terms: list[str], doc_texts: dict[str, str], max_chars: int = 220) -> str:
    """Return a short snippet around first matched query term in a document."""
    text = (doc_texts.get(doc_id) or "").strip()
    if not text:
        return "[no text available for this document]"

    start_index = None
    match_len = 0
    lowered = text.lower()
    for term in sorted(terms, key=len, reverse=True):
        idx = lowered.find(term.lower())
        if idx != -1:
            start_index = idx
            match_len = len(term)
            break

    if start_index is None:
        snippet = text[:max_chars]
        if len(text) > max_chars:
            snippet += "..."
        return _highlight_text(snippet, terms)

    left = max(0, start_index - 80)
    right = min(len(text), start_index + match_len + 120)
    snippet = text[left:right]
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet = snippet + "..."

    return _highlight_text(snippet, terms)


def _print_debug_hits(
    ranked: list[tuple[str, float]],
    query: str,
    query_preprocessor: PipelineQueryPreprocessor | None,
    doc_texts: dict[str, str],
    max_docs: int = 3,
) -> None:
    """Print debug snippets from top-ranked documents with highlighted query terms."""
    if not ranked:
        return

    terms = _extract_query_terms(query, query_preprocessor)
    if not terms:
        print("\n[debug] No highlightable terms extracted from query.")
        return

    print("\n[debug] Document snippet check (highlighted terms in [[...]]):")
    for rank, (doc_id, _) in enumerate(ranked[:max_docs], start=1):
        snippet = _build_debug_snippet(doc_id, terms, doc_texts)
        print(f"  {rank}. {doc_id}")
        print(f"     {snippet}")


def _ask_search_method() -> str:
    """Ask user to choose between TF-IDF and Boolean search."""
    print("\n" + "=" * 60)
    print("SEARCH METHOD")
    print("=" * 60)
    print("1. TF-IDF (ranked retrieval with similarity scores)")
    print("2. Boolean (exact term matching with AND/OR/NOT operators)")
    print("=" * 60)
    
    while True:
        choice = input("Choose search method (1 or 2): ").strip()
        if choice in ("1", "2"):
            return "tfidf" if choice == "1" else "boolean"
        print("Invalid choice. Please enter 1 or 2.")


def _build_boolean_index_from_tfidf_docs(index: InvertedIndex, docs: dict[str, str], pipeline_name: str) -> BooleanIndex:
    """Convert TF-IDF index to Boolean index when user wants Boolean search."""
    preprocessor = Preprocessor()
    boolean_index = BooleanIndex()
    boolean_index.build(docs, preprocessor)
    return boolean_index


def run_interactive_query_loop(index_file: str, pipeline: str, doc_texts: dict[str, str] | None = None) -> int:
    """
    Run an interactive query loop allowing user to choose search method and submit multiple queries.
    
    Args:
        index_file: Path to the index JSON file
        pipeline: Pipeline name for query preprocessing
        doc_texts: Optional dict of doc_id -> text for Boolean search (auto-loaded if not provided)
    
    Returns:
        0 on success, 1 on error
    """
    index_path = Path(index_file)
    
    if not index_path.exists():
        print(f"Error: Index file not found: {index_path}")
        return 1
    
    # Load metadata and document texts from crawler data
    metadata = {}
    debug_doc_texts: dict[str, str] = dict(doc_texts or {})
    root = Path(__file__).resolve().parent.parent
    crawler_data_path = root / "data" / "crawler" / "crawled_pages.json"
    
    if crawler_data_path.exists():
        try:
            with crawler_data_path.open("r", encoding="utf-8") as f:
                # crawled_pages.json is JSONL (newline-delimited JSON)
                for line in f:
                    try:
                        doc = json.loads(line)
                        url = doc.get("url")
                        if url:
                            metadata[url] = {
                                "title": doc.get("title", "No title"),
                                "author": doc.get("author", ""),
                            }
                            article_text = doc.get("article_text")
                            if isinstance(article_text, str) and article_text.strip():
                                debug_doc_texts[url] = article_text
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Note: Could not load document metadata ({e})")
    
    # Load TF-IDF index
    try:
        with index_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        if "index" in loaded and isinstance(loaded["index"], dict):
            index_data = loaded["index"]
            meta = loaded.get("meta", {}) if isinstance(loaded.get("meta", {}), dict) else {}
        else:
            index_data = loaded
            meta = {}
        
        tfidf_index = InvertedIndex.from_dict(index_data)
        resolved_pipeline = str(meta.get("pipeline") or pipeline)
    except Exception as e:
        print(f"Error loading index: {e}")
        return 1
    
    print(f"\n✓ Loaded index: {index_path.name}")
    print(f"  Pipeline: {resolved_pipeline}")
    print(f"  Documents: {tfidf_index.num_docs}, Terms: {len(tfidf_index.postings)}")
    if metadata:
        print(f"  Metadata: {len(metadata)} documents loaded")
    
    # Ask which search method
    search_method = _ask_search_method()
    
    # Create preprocessors
    query_preprocessor = PipelineQueryPreprocessor(resolved_pipeline)
    
    if search_method == "boolean":
        boolean_index = BooleanIndex()
        if doc_texts:
            preprocessor = Preprocessor()
            boolean_index.build(doc_texts, preprocessor)
        else:
            # Try to reconstruct from TF-IDF index postings
            preprocessor = Preprocessor()
            documents = {doc_id: f"[document with terms]" for doc_id in tfidf_index.doc_term_freqs.keys()}
            boolean_index.build(documents, preprocessor)
        
        boolean_scorer = BooleanScorer(boolean_index, preprocessor)
    else:
        tfidf_scorer = CosineScorer(tfidf_index, query_preprocessor)
    
    # Query loop
    print("\n" + "=" * 60)
    print("QUERY INTERFACE")
    print("=" * 60)
    print("Enter queries to search. Type 'quit' or 'exit' to finish.")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n📝 Enter query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                return 0
            
            print(f"\n🔍 Searching for: '{query}'")
            
            if search_method == "tfidf":
                ranked = tfidf_scorer.search(query)
                print(f"\nResults ({len(ranked)} found):")
            else:  # boolean
                ranked = boolean_scorer.search(query)
                print(f"\nResults ({len(ranked)} found):")
            
            print(_format_results(ranked, search_method, metadata, max_display=10))
            _print_debug_hits(
                ranked=ranked,
                query=query,
                query_preprocessor=query_preprocessor,
                doc_texts=debug_doc_texts,
            )
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            return 0
        except Exception as e:
            print(f"Error processing query: {e}")
            continue
