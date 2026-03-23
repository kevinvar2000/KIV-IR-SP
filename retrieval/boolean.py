from typing import Dict, List, Set, Tuple


class BooleanIndex:
    """Term -> set(doc_id) index for Boolean retrieval."""

    def __init__(self) -> None:
        self.postings: Dict[str, Set[str]] = {}
        self.documents: Dict[str, str] = {}

    def build(self, documents: Dict[str, str], preprocessor) -> None:
        self.documents = documents
        for doc_id, text in documents.items():
            seen_terms: Set[str] = set(preprocessor.tokenize(text))
            for term in seen_terms:
                if term not in self.postings:
                    self.postings[term] = set()
                self.postings[term].add(doc_id)


class BooleanScorer:
    """Basic Boolean retrieval supporting AND/OR/NOT (left-to-right evaluation)."""

    def __init__(self, index: BooleanIndex, preprocessor) -> None:
        self.index = index
        self.preprocessor = preprocessor

    def _tokenize_query(self, query_text: str) -> List[str]:
        raw_parts = query_text.strip().split()
        tokens: List[str] = []
        for part in raw_parts:
            upper = part.upper()
            if upper in {"AND", "OR", "NOT"}:
                tokens.append(upper)
            else:
                normalized = self.preprocessor.tokenize(part)
                if normalized:
                    tokens.extend(normalized)
        return tokens

    def search(self, query_text: str) -> List[Tuple[str, float]]:
        all_docs = set(self.index.documents.keys())
        tokens = self._tokenize_query(query_text)
        if not tokens:
            return []

        current: Set[str] | None = None
        op = "OR"
        negate_next = False

        for token in tokens:
            if token in {"AND", "OR"}:
                op = token
                continue
            if token == "NOT":
                negate_next = True
                continue

            docs = set(self.index.postings.get(token, set()))
            if negate_next:
                docs = all_docs - docs
                negate_next = False

            if current is None:
                current = docs
            elif op == "AND":
                current = current & docs
            else:
                current = current | docs

        result_docs = sorted(current or [])
        return [(doc_id, 1.0) for doc_id in result_docs]
