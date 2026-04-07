"""Tokenizer implementations and token data structures for text preprocessing."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class TokenType(Enum):
    """Enumeration of token categories recognized by regex tokenizer."""

    URL = 1
    DATE = 2
    NUMBER = 3
    WORD = 4
    TAG = 5
    PUNCT = 6

@dataclass
class Token:
    """Single token with normalized text and source position metadata."""

    processed_form: str
    position: int
    length: int
    token_type: TokenType = TokenType.WORD

    def __repr__(self):
        """Return processed token form for concise debug output."""
        return self.processed_form



class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, document: str) -> list[Token]:
        """Split input document into ordered token objects."""
        raise NotImplementedError() 


class SplitTokenizer(Tokenizer):
    """Tokenizer that splits text by a single delimiter character."""

    def __init__(self, split_char: str):
        """Initialize tokenizer with delimiter character."""
        self.split_char = split_char

    def tokenize(self, document: str) -> list[Token]:
        """Split document by configured delimiter and return token sequence."""
        tokens = []
        position = 0
        for word in document.split(self.split_char):
            token = Token(word, position, len(word))
            tokens.append(token)
            position += len(word) + 1
        return tokens

class RegexMatchTokenizer(Tokenizer):
    """Regex tokenizer with named groups for URL/date/number/word/tag/punctuation."""


    # Keep more specific patterns first so they are not split by generic ones.
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    date_pattern = r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b'
    num_pattern = r'\d+(?:[.,]\d+)*'  # matches numbers like 123, 123.123, 123,123
    word_pattern = r'\w+'  # matches words
    html_tag_pattern = r'<[^>]+>'  # matches html tags
    punctuation_pattern = r'[^\w\s]+'  # matches punctuation
    default_pattern = (
        f'(?P<URL>{url_pattern})|'
        f'(?P<DATE>{date_pattern})|'
        f'(?P<NUMBER>{num_pattern})|'
        f'(?P<WORD>{word_pattern})|'
        f'(?P<TAG>{html_tag_pattern})|'
        f'(?P<PUNCT>{punctuation_pattern})'
    )
    # default_pattern = r'(\d+[.,](\d+)?)|([\w]+)|(<.*?>)|([^\w\s]+)'

    def __init__(self, pattern: str=default_pattern):
        """Compile regex pattern used for token extraction."""
        self.pattern = pattern
        self._compiled = re.compile(pattern, re.UNICODE)

    def tokenize(self, document: str) -> list[Token]:
        """Tokenize document using regex named groups and position metadata."""
        tokens = []
        for match in re.finditer(self._compiled, document):
            group_name = match.lastgroup or "WORD"
            token = Token(
                match.group(),
                match.start(),
                match.end() - match.start(),
                TokenType[group_name],
            )
            tokens.append(token)
        return tokens

if __name__ == '__main__':
    document = "Hello, world! This is a test."
    document = 'příliš žluťoučký kůň úpěl ďábelské ódy. 20.25'
    tokenizer = SplitTokenizer(" ")
    tokens = tokenizer.tokenize(document)
    print(tokens)
    tokenizer = RegexMatchTokenizer()
    tokens = tokenizer.tokenize(document)
    print(tokens)