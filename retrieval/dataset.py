"""Collection parsing and lightweight tokenization for retrieval workflows."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


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
        """Tokenize text into lowercased word terms."""
        return [token.lower() for token in self.token_pattern.findall(text)]


class CollectionParser:
    """Parses files containing lines like `d1: ...` and `q1: ...`."""

    @staticmethod
    def _looks_like_vocab_line(line: str) -> bool:
        """Detect `term count` lines that likely come from vocab outputs."""
        parts = line.split()
        return len(parts) == 2 and parts[1].isdigit()

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
                if self._looks_like_vocab_line(stripped):
                    raise ValueError(
                        f"Invalid retrieval input format in {path.name}: '{line}'. "
                        "This looks like a preprocessing vocabulary file ('term count'). "
                        "Retrieval expects collection lines such as 'd1: ...' and 'q1: ...'."
                    )
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
