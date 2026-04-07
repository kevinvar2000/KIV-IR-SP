"""Token-level preprocessing primitives and configurable pipeline composition."""

from abc import ABC, abstractmethod
from functools import lru_cache
import unicodedata

import simplemma

from .tokenizer import Token, TokenType
from .language_config import get_stopwords, get_stem_suffixes, normalize_language_code


@lru_cache(maxsize=200_000)
def _strip_diacritics_cached(text: str) -> str:
    """Remove diacritic marks with memoization for repeated tokens."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


@lru_cache(maxsize=200_000)
def _simplemma_lemmatize_cached(word: str, lang: str, greedy: bool = False) -> str:
    """Memoized simplemma lemmatization helper."""
    return simplemma.lemmatize(word, lang=lang, greedy=greedy)


class TokenPreprocessor(ABC):
    """Abstract base for token-level preprocessing operations."""

    @abstractmethod
    def preprocess(self, token: Token, document: str) -> Token:
        """Transform a single token and optionally drop it by returning None."""
        raise NotImplementedError()

    def preprocess_all(self, tokens: list[Token], document: str) -> list[Token]:
        """Apply per-token preprocessing to full token sequence."""
        return [self.preprocess(token, document) for token in tokens]


class LowercasePreprocessor(TokenPreprocessor):
    """Converts token text to lowercase."""

    def preprocess(self, token: Token, document: str) -> Token:
        """Lowercase token text in place."""
        token.processed_form = token.processed_form.lower()
        return token


class RemoveDiacriticsPreprocessor(TokenPreprocessor):
    """Strips diacritics (accents) from token text.  e.g. příliš -> prilis"""

    def preprocess(self, token: Token, document: str) -> Token:
        """Remove accents from token text."""
        token.processed_form = _strip_diacritics_cached(token.processed_form)
        return token


class RemoveTokenTypesPreprocessor(TokenPreprocessor):
    """Removes tokens whose type is in the forbidden set (e.g. PUNCT, TAG)."""

    def __init__(self, forbidden_types: set[TokenType]):
        """Store token types that should be filtered from the stream."""
        self.forbidden_types = forbidden_types

    def preprocess(self, token: Token, document: str) -> Token | None:
        """Drop token when its type is forbidden."""
        if token.token_type in self.forbidden_types:
            return None
        return token

    def preprocess_all(self, tokens: list[Token], document: str) -> list[Token]:
        """Filter token list by applying token-type rule."""
        return [t for t in tokens if self.preprocess(t, document) is not None]


class MinLengthPreprocessor(TokenPreprocessor):
    """Drops tokens shorter than min_length characters."""

    def __init__(self, min_length: int = 2):
        """Configure minimum accepted token length."""
        self.min_length = min_length

    def preprocess(self, token: Token, document: str) -> Token | None:
        """Drop token when processed form is shorter than minimum length."""
        if len(token.processed_form) < self.min_length:
            return None
        return token

    def preprocess_all(self, tokens: list[Token], document: str) -> list[Token]:
        """Filter token list using minimum-length rule."""
        return [t for t in tokens if self.preprocess(t, document) is not None]


class StopwordPreprocessor(TokenPreprocessor):
    """Removes stopwords for the selected language from the token stream."""

    def __init__(self, stopwords: set[str] | None = None, language: str = "cs"):
        """Initialize stopword sets for exact and diacritics-stripped matching."""
        self._stopwords = stopwords or get_stopwords(language)
        self._stopwords_ascii = {_strip_diacritics_cached(token) for token in self._stopwords}

    def preprocess(self, token: Token, document: str) -> Token | None:
        """Drop token when it matches exact or accent-insensitive stopwords."""
        processed = token.processed_form
        if processed in self._stopwords:
            return None
        if _strip_diacritics_cached(processed) in self._stopwords_ascii:
            return None
        return token

    def preprocess_all(self, tokens: list[Token], document: str) -> list[Token]:
        """Filter token list using stopword checks."""
        return [t for t in tokens if self.preprocess(t, document) is not None]


class StemmingPreprocessor(TokenPreprocessor):
    """Stemming for Czech tokens with language-aware fallback lemmatization."""

    def __init__(self, language: str = "cs"):
        """Initialize stemming strategy and caches for a selected language."""
        self.lang = normalize_language_code(language)
        self._cs_suffixes = tuple(sorted(set(get_stem_suffixes("cs")), key=len, reverse=True))
        self._stem_cache: dict[str, str] = {}

    def _fallback_stem_cs(self, word: str) -> str:
        """Apply conservative Czech suffix stripping."""
        # Light-weight Czech suffix stripper.
        # This is intentionally conservative to avoid over-stemming short words.
        if len(word) < 5:
            return word

        for suffix in self._cs_suffixes:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        return word

    def preprocess(self, token: Token, document: str) -> Token:
        """Stem word tokens with cached language-specific strategy."""
        if token.token_type == TokenType.WORD:
            current = token.processed_form
            cached = self._stem_cache.get(current)
            if cached is None:
                if self.lang == "cs":
                    cached = self._fallback_stem_cs(current)
                else:
                    cached = _simplemma_lemmatize_cached(current, self.lang, True)
                self._stem_cache[current] = cached
            token.processed_form = cached
        return token


class LemmatizationPreprocessor(TokenPreprocessor):
    """Lemmatization using simplemma for Czech, Slovak, and English."""

    def __init__(self, language: str = "cs"):
        """Initialize lemmatizer cache for a selected language."""
        self.lang = normalize_language_code(language)
        self._lemma_cache: dict[str, str] = {}

    def preprocess(self, token: Token, document: str) -> Token:
        """Lemmatize word tokens and reuse cached results."""
        if token.token_type == TokenType.WORD:
            current = token.processed_form
            cached = self._lemma_cache.get(current)
            if cached is None:
                cached = _simplemma_lemmatize_cached(current, self.lang, False)
                self._lemma_cache[current] = cached
            token.processed_form = cached
        return token


class PreprocessingPipeline:
    """Sequentially applies a list of preprocessors to a token list."""

    def __init__(self, preprocessors: list[TokenPreprocessor]):
        """Store ordered preprocessors to execute."""
        self.preprocessors = preprocessors

    def preprocess(self, tokens: list[Token], document: str) -> list[Token]:
        """Run all preprocessors sequentially over token list."""
        for preprocessor in self.preprocessors:
            tokens = preprocessor.preprocess_all(tokens, document)
        return tokens

