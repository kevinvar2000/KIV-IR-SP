"""Interactive query interface for TF-IDF and Boolean search."""

import json
import re
from pathlib import Path
import interactive_config as ui

from preprocessing.config import PIPELINE_NAMES, build_pipelines
from preprocessing.language_config import normalize_language_code
from preprocessing.tokenizer import RegexMatchTokenizer

from .tfidf import CosineScorer, InvertedIndex
from .boolean import BooleanIndex, BooleanScorer
from .dataset import Preprocessor

def _is_nav_or_exit_command(value: str) -> bool:
    """Return True when input requests leaving the current interactive loop."""
    return value.strip().lower() in ui.NAV_OR_EXIT_COMMANDS


class PipelineQueryPreprocessor:
    """Apply the same preprocessing pipeline to queries as used for index generation."""

    @staticmethod
    def _resolve_pipeline_and_language(pipeline_name: str) -> tuple[str, str]:
        """Resolve pipeline base name and language suffix from persisted metadata."""
        normalized = pipeline_name.strip()
        if normalized in PIPELINE_NAMES:
            return normalized, "cs"

        for language_code in ("cs", "sk", "en"):
            suffix = f"_{language_code}"
            if normalized.endswith(suffix):
                candidate = normalized[: -len(suffix)]
                if candidate in PIPELINE_NAMES:
                    return candidate, normalize_language_code(language_code)

        raise ValueError(f"Unknown preprocessing pipeline for query mode: {pipeline_name}")

    def __init__(self, pipeline_name: str):
        """Initialize query tokenizer and matching preprocessing pipeline."""
        resolved_pipeline, resolved_language = self._resolve_pipeline_and_language(pipeline_name)
        pipelines = build_pipelines(resolved_language)
        self.pipeline = pipelines[resolved_pipeline]
        self.tokenizer = RegexMatchTokenizer()

    def tokenize(self, text: str) -> list[str]:
        """Tokenize and normalize query text using configured pipeline."""
        tokens = self.tokenizer.tokenize(text)
        tokens = self.pipeline.preprocess(tokens, text)
        return [token.processed_form for token in tokens if token.processed_form]


def _format_results(results: list[tuple[str, float]], method: str, metadata: dict[str, dict] | None = None, max_display: int = 10) -> str:
    """Format search results nicely with rank | score | title | link."""
    if not results:
        return ui.QUERY_FORMAT_NO_MATCH

    lines = []
    limited = results[:max_display]
    
    # Header
    if method == "tfidf":
        lines.append(ui.DIVIDER)
        lines.append(ui.QUERY_FORMAT_HEADER_TFIDF)
        lines.append("-" * 60)
    else:
        lines.append(ui.DIVIDER)
        lines.append(ui.QUERY_FORMAT_HEADER_BOOLEAN)
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
    lines.append(ui.DIVIDER)
    
    if len(results) > max_display:
        lines.append(ui.QUERY_FORMAT_MORE_RESULTS.format(count=len(results) - max_display))
    
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


def _ask_search_method() -> str | None:
    """Ask user to choose between TF-IDF and Boolean search."""
    print("\n" + ui.DIVIDER)
    print(ui.QUERY_METHOD_TITLE)
    print(ui.DIVIDER)
    print(ui.QUERY_METHOD_LINE_1)
    print(ui.QUERY_METHOD_LINE_2)
    print(ui.QUERY_METHOD_LINE_0)
    print(ui.DIVIDER)
    
    while True:
        choice = input(ui.PROMPT_QUERY_METHOD).strip().lower()
        if choice in ("1", "2"):
            return "tfidf" if choice == "1" else "boolean"
        if choice in ui.HOME_COMMANDS:
            return None
        if choice in ui.EXIT_COMMANDS:
            return ui.APP_EXIT_SIGNAL
        print(ui.QUERY_METHOD_INVALID)


def _build_boolean_index_from_tfidf_docs(index: InvertedIndex) -> BooleanIndex:
    """Convert loaded TF-IDF structures into a Boolean index with matching term space."""
    boolean_index = BooleanIndex()
    boolean_index.documents = {doc_id: "" for doc_id in index.doc_term_freqs.keys()}
    boolean_index.postings = {
        term: set(posting.keys())
        for term, posting in index.postings.items()
    }
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
        print(ui.QUERY_INDEX_NOT_FOUND.format(path=index_path))
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
            print(ui.QUERY_METADATA_LOAD_NOTE.format(error=e))
    
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
        print(ui.QUERY_LOAD_INDEX_ERROR.format(error=e))
        return 1
    
    print(ui.QUERY_LOADED_INDEX.format(name=index_path.name))
    print(ui.QUERY_PIPELINE.format(pipeline=resolved_pipeline))
    print(ui.QUERY_DOCS_TERMS.format(docs=tfidf_index.num_docs, terms=len(tfidf_index.postings)))
    if metadata:
        print(ui.QUERY_METADATA_COUNT.format(count=len(metadata)))
    
    # Create preprocessors
    query_preprocessor = PipelineQueryPreprocessor(resolved_pipeline)
    boolean_index = _build_boolean_index_from_tfidf_docs(tfidf_index)
    boolean_scorer = BooleanScorer(boolean_index, query_preprocessor, debug=False)
    tfidf_scorer = CosineScorer(tfidf_index, query_preprocessor)
    
    # Query loop
    print("\n" + ui.DIVIDER)
    print(ui.QUERY_INTERFACE_TITLE)
    print(ui.DIVIDER)
    print(ui.QUERY_INTERFACE_HELP)
    print(ui.DIVIDER)

    search_method = _ask_search_method()
    if search_method == ui.APP_EXIT_SIGNAL:
        return ui.APP_EXIT_CODE
    if search_method is None:
        print(ui.QUERY_INTERFACE_RETURNING)
        return 0

    print(ui.QUERY_INTERFACE_SELECTED_METHOD.format(method=search_method))
    
    while True:
        try:
            query = input(ui.PROMPT_ENTER_QUERY).strip()
            
            if not query:
                continue
            
            if _is_nav_or_exit_command(query):
                print(ui.QUERY_INTERFACE_RETURNING)
                return 0

            print(ui.QUERY_SEARCHING.format(query=query))
            
            if search_method == "tfidf":
                ranked = tfidf_scorer.search(query)
                print(ui.QUERY_TOTAL_FOUND.format(count=len(ranked)))
                print(ui.QUERY_RESULTS_FOUND.format(count=len(ranked)))
            else:  # boolean
                ranked = boolean_scorer.search(query)
                debug_data = boolean_scorer.last_debug
                print(ui.QUERY_TOTAL_FOUND.format(count=len(ranked)))
                print(ui.QUERY_RESULTS_FOUND.format(count=len(ranked)))
                print(f"[debug][boolean] infix={debug_data.get('infix_tokens', [])}")
                print(f"[debug][boolean] postfix={debug_data.get('postfix_tokens', [])}")
            
            print(_format_results(ranked, search_method, metadata, max_display=10))
            _print_debug_hits(
                ranked=ranked,
                query=query,
                query_preprocessor=query_preprocessor,
                doc_texts=debug_doc_texts,
            )
        
        except KeyboardInterrupt:
            print(ui.QUERY_INTERRUPTED)
            return 0
        except Exception as e:
            print(ui.QUERY_PROCESSING_ERROR.format(error=e))
            continue
