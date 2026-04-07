"""Boolean retrieval index and scorer with infix query parsing."""

import re
from typing import Dict, List, Set, Tuple


class BooleanIndex:
    """Term -> set(doc_id) index for Boolean retrieval."""

    def __init__(self) -> None:
        """Initialize empty postings and document collections."""
        self.postings: Dict[str, Set[str]] = {}
        self.documents: Dict[str, str] = {}

    def build(self, documents: Dict[str, str], preprocessor) -> None:
        """Build unique-term postings lists for each document."""
        self.documents = documents
        for doc_id, text in documents.items():
            seen_terms: Set[str] = set(preprocessor.tokenize(text))
            for term in seen_terms:
                if term not in self.postings:
                    self.postings[term] = set()
                self.postings[term].add(doc_id)


class BooleanScorer:
    """Boolean retrieval supporting AND/OR/NOT with parentheses and precedence."""

    _OPERATORS = {"AND", "OR", "NOT"}
    _PRECEDENCE = {"OR": 1, "AND": 2, "NOT": 3}
    _ASSOCIATIVITY = {"OR": "left", "AND": "left", "NOT": "right"}
    _TOKEN_RE = re.compile(r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+", flags=re.IGNORECASE)

    def __init__(self, index: BooleanIndex, preprocessor, debug: bool = False) -> None:
        """Initialize scorer with index, query preprocessor, and debug mode."""
        self.index = index
        self.preprocessor = preprocessor
        self.debug = debug
        self.last_debug: Dict[str, object] = {}

    @staticmethod
    def _is_operand(token: str) -> bool:
        """Return True when token is a query term, not an operator or parenthesis."""
        return token not in {"AND", "OR", "NOT", "(", ")"}

    @staticmethod
    def _insert_implicit_or(tokens: List[str]) -> List[str]:
        """Insert implicit OR between adjacent operands/groups when omitted by user."""
        if not tokens:
            return []

        expanded: List[str] = [tokens[0]]
        for token in tokens[1:]:
            prev = expanded[-1]
            needs_or = (
                (BooleanScorer._is_operand(prev) or prev == ")")
                and (BooleanScorer._is_operand(token) or token in {"(", "NOT"})
            )
            if needs_or:
                expanded.append("OR")
            expanded.append(token)
        return expanded

    def _tokenize_query(self, query_text: str) -> List[str]:
        """Tokenize Boolean query into normalized infix token stream."""
        raw_parts = self._TOKEN_RE.findall(query_text.strip())
        tokens: List[str] = []

        for part in raw_parts:
            upper = part.upper()
            if upper in {"AND", "OR", "NOT"}:
                tokens.append(upper)
            elif part in {"(", ")"}:
                tokens.append(part)
            else:
                normalized = self.preprocessor.tokenize(part)
                if normalized:
                    if len(normalized) == 1:
                        tokens.append(normalized[0])
                    else:
                        tokens.append("(")
                        for i, token in enumerate(normalized):
                            if i > 0:
                                tokens.append("OR")
                            tokens.append(token)
                        tokens.append(")")

        return self._insert_implicit_or(tokens)

    def _infix_to_postfix(self, tokens: List[str]) -> List[str]:
        """Convert infix Boolean expression to postfix using shunting-yard logic."""
        output: List[str] = []
        stack: List[str] = []

        for token in tokens:
            if self._is_operand(token):
                output.append(token)
                continue

            if token == "(":
                stack.append(token)
                continue

            if token == ")":
                while stack and stack[-1] != "(":
                    output.append(stack.pop())
                if not stack:
                    raise ValueError("Mismatched parentheses in Boolean query.")
                stack.pop()
                continue

            while stack and stack[-1] in self._OPERATORS:
                top = stack[-1]
                assoc = self._ASSOCIATIVITY[token]
                if (
                    (assoc == "left" and self._PRECEDENCE[token] <= self._PRECEDENCE[top])
                    or (assoc == "right" and self._PRECEDENCE[token] < self._PRECEDENCE[top])
                ):
                    output.append(stack.pop())
                else:
                    break

            stack.append(token)

        while stack:
            top = stack.pop()
            if top in {"(", ")"}:
                raise ValueError("Mismatched parentheses in Boolean query.")
            output.append(top)

        return output

    def _evaluate_postfix(self, postfix_tokens: List[str]) -> Set[str]:
        """Evaluate postfix Boolean query and return matching document identifiers."""
        all_docs = set(self.index.documents.keys())
        stack: List[Set[str]] = []

        for token in postfix_tokens:
            if self._is_operand(token):
                stack.append(set(self.index.postings.get(token, set())))
                continue

            if token == "NOT":
                if not stack:
                    raise ValueError("Malformed Boolean query near NOT.")
                operand = stack.pop()
                stack.append(all_docs - operand)
                continue

            if len(stack) < 2:
                raise ValueError(f"Malformed Boolean query near {token}.")

            right = stack.pop()
            left = stack.pop()
            if token == "AND":
                stack.append(left & right)
            elif token == "OR":
                stack.append(left | right)
            else:
                raise ValueError(f"Unsupported Boolean operator: {token}")

        if len(stack) != 1:
            raise ValueError("Malformed Boolean query.")
        return stack[0]

    def search(self, query_text: str) -> List[Tuple[str, float]]:
        """Run full Boolean retrieval pipeline and return sorted matches."""
        infix_tokens = self._tokenize_query(query_text)
        if not infix_tokens:
            self.last_debug = {
                "infix_tokens": [],
                "postfix_tokens": [],
                "hit_count": 0,
            }
            return []

        postfix_tokens = self._infix_to_postfix(infix_tokens)
        result_set = self._evaluate_postfix(postfix_tokens)
        result_docs = sorted(result_set)

        self.last_debug = {
            "infix_tokens": infix_tokens,
            "postfix_tokens": postfix_tokens,
            "hit_count": len(result_docs),
        }
        if self.debug:
            print(f"[debug][boolean] infix={infix_tokens}")
            print(f"[debug][boolean] postfix={postfix_tokens}")
            print(f"[debug][boolean] hits={len(result_docs)}")

        return [(doc_id, 1.0) for doc_id in result_docs]
