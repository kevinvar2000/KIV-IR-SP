from abc import ABC, abstractmethod
import unicodedata

import simplemma

try:
    from .tokenizer import Token, TokenType
    from .language_config import get_stopwords, normalize_language_code
except ImportError:
    from tokenizer import Token, TokenType
    from language_config import get_stopwords, normalize_language_code


class TokenPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, token: Token, document: str) -> Token:
        raise NotImplementedError()

    def preprocess_all(self, tokens: list[Token], document: str) -> list[Token]:
        return [self.preprocess(token, document) for token in tokens]


class LowercasePreprocessor(TokenPreprocessor):
    """Converts token text to lowercase."""

    def preprocess(self, token: Token, document: str) -> Token:
        token.processed_form = token.processed_form.lower()
        return token


class RemoveDiacriticsPreprocessor(TokenPreprocessor):
    """Strips diacritics (accents) from token text.  e.g. příliš -> prilis"""

    def preprocess(self, token: Token, document: str) -> Token:
        nfkd = unicodedata.normalize("NFKD", token.processed_form)
        token.processed_form = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
        return token


class RemoveTokenTypesPreprocessor(TokenPreprocessor):
    """Removes tokens whose type is in the forbidden set (e.g. PUNCT, TAG)."""

    def __init__(self, forbidden_types: set[TokenType]):
        self.forbidden_types = forbidden_types

    def preprocess(self, token: Token, document: str) -> Token | None:
        if token.token_type in self.forbidden_types:
            return None
        return token

    def preprocess_all(self, tokens: list[Token], document: str) -> list[Token]:
        return [t for t in tokens if self.preprocess(t, document) is not None]


class MinLengthPreprocessor(TokenPreprocessor):
    """Drops tokens shorter than min_length characters."""

    def __init__(self, min_length: int = 2):
        self.min_length = min_length

    def preprocess(self, token: Token, document: str) -> Token | None:
        if len(token.processed_form) < self.min_length:
            return None
        return token

    def preprocess_all(self, tokens: list[Token], document: str) -> list[Token]:
        return [t for t in tokens if self.preprocess(t, document) is not None]


class StopwordPreprocessor(TokenPreprocessor):
    """Removes stopwords for the selected language from the token stream."""

    def __init__(self, stopwords: set[str] | None = None, language: str = "cs"):
        self._stopwords = stopwords or get_stopwords(language)
        self._stopwords_ascii = {self._strip_diacritics(token) for token in self._stopwords}

    @staticmethod
    def _strip_diacritics(text: str) -> str:
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

    def preprocess(self, token: Token, document: str) -> Token | None:
        processed = token.processed_form
        if processed in self._stopwords:
            return None
        if self._strip_diacritics(processed) in self._stopwords_ascii:
            return None
        return token

    def preprocess_all(self, tokens: list[Token], document: str) -> list[Token]:
        return [t for t in tokens if self.preprocess(t, document) is not None]


class StemmingPreprocessor(TokenPreprocessor):
    """Stemming for Czech tokens with language-aware fallback lemmatization."""

    def __init__(self, language: str = "cs"):
        self.lang = normalize_language_code(language)

    def _fallback_stem_cs(self, word: str) -> str:
        # Light-weight Czech suffix stripper.
        # This is intentionally conservative to avoid over-stemming short words.
        if len(word) < 5:
            return word

        endings = (
            "ami", "emi", "ovi", "ách", "ích", "ého", "ěho", "ému", "ěmu",
            "ové", "ovi", "ého", "ěmi", "ami", "emi", "kami", "kám", "kou",
            "ami", "ové", "ové", "ost", "ostí", "osti", "ování", "ování",
            "ové", "ích", "ách", "ům", "em", "om", "ou", "mi", "ho", "mu",
            "ty", "ti", "te", "ta", "tu", "ch", "ou", "ů", "y", "a", "e", "i",
        )
        for suffix in sorted(set(endings), key=len, reverse=True):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        return word

    def preprocess(self, token: Token, document: str) -> Token:
        if token.token_type == TokenType.WORD:
            if self.lang == "cs":
                token.processed_form = self._fallback_stem_cs(token.processed_form)
            else:
                token.processed_form = simplemma.lemmatize(token.processed_form, lang=self.lang, greedy=True)
        return token


class LemmatizationPreprocessor(TokenPreprocessor):
    """Lemmatization using simplemma for Czech, Slovak, and English."""

    def __init__(self, language: str = "cs"):
        self.lang = normalize_language_code(language)

    def preprocess(self, token: Token, document: str) -> Token:
        if token.token_type == TokenType.WORD:
            token.processed_form = simplemma.lemmatize(token.processed_form, lang=self.lang)
        return token


class PreprocessingPipeline:
    """Sequentially applies a list of preprocessors to a token list."""

    def __init__(self, preprocessors: list[TokenPreprocessor]):
        self.preprocessors = preprocessors

    def preprocess(self, tokens: list[Token], document: str) -> list[Token]:
        for preprocessor in self.preprocessors:
            tokens = preprocessor.preprocess_all(tokens, document)
        return tokens

