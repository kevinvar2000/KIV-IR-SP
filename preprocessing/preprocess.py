from abc import ABC, abstractmethod
import unicodedata

import simplemma

try:
    from .tokenizer import Token, TokenType
except ImportError:
    from tokenizer import Token, TokenType

# Czech stopwords – standard list used in Czech NLP / IR
CZECH_STOPWORDS: set[str] = {
    "a", "aby", "aj", "ale", "ani", "aniž", "ano", "asi", "aspoň", "atd",
    "atp", "az", "ačkoli", "až", "bez", "beze", "blízko", "bohužel", "brzo",
    "brzy", "buď", "budu", "by", "byl", "byla", "byli", "bylo", "byly",
    "bys", "být", "během", "co", "což", "což", "cz", "či", "článek",
    "článku", "články", "další", "dnes", "do", "dokonce", "dokud", "dost",
    "dosud", "doufám", "dva", "dvě", "dál", "dále", "děkovat", "děkuji",
    "ho", "hodně", "i", "jak", "jakmile", "jako", "jakož", "jaký", "je",
    "jeden", "jedna", "jednak", "jedno", "jednou", "jedny", "jeho", "jej",
    "její", "jejich", "jemu", "jen", "jenž", "jestli", "jestliže", "ještě",
    "jež", "ji", "jich", "jimi", "jinak", "jiné", "jiní", "jiný", "již",
    "jsi", "jsme", "jsou", "jste", "já", "jí", "jím", "k", "kam", "kde",
    "kdo", "kdy", "když", "ke", "kolem", "kolik", "kromě", "krátce", "krátký",
    "která", "které", "kteří", "který", "kvůli", "má", "mají", "málo", "mě",
    "mezi", "mi", "mne", "mnou", "mně", "moc", "mohl", "mohou", "moje",
    "mojí", "možná", "muj", "musí", "my", "myslím", "má", "mám", "máte",
    "mít", "mě", "můj", "může", "na", "nad", "nade", "nam", "naproti",
    "nás", "naše", "naši", "ne", "neboť", "nebo", "nebyl", "nebyla", "nebyli",
    "nebylo", "nebyly", "nechť", "nedělá", "nedělají", "nedělám", "nedělate",
    "neg", "nej", "nejsou", "někde", "někdo", "některá", "některé", "některý",
    "nemají", "nemá", "neměl", "nemůže", "nes", "nesmi", "nesmí", "než",
    "nic", "nichž", "ní", "ním", "nimi", "no", "nové", "nový", "nás",
    "nám", "ná", "něco", "nějak", "nějaký", "němu", "němuž", "o", "od",
    "ode", "on", "ona", "oni", "ono", "ony", "ostatně", "pak", "pan", "paní",
    "po", "pod", "podle", "pokud", "pouze", "potom", "poté", "pořád",
    "prav", "pravé", "pro", "proč", "prostě", "prosím", "proto", "protože",
    "první", "před", "přede", "přes", "přese", "přesto", "při", "přičemž",
    "re", "rovněž", "s", "sám", "se", "si", "sice", "skoro", "smí", "smějí",
    "snad", "spolu", "sta", "strana", "své", "svého", "svou", "svůj", "svým",
    "svými", "sa", "sám", "své", "ta", "tak", "také", "takže", "tam",
    "tamhle", "tamhleto", "tamto", "tato", "tedy", "ten", "tento", "ti",
    "tím", "tímto", "to", "toho", "tohle", "tomu", "tomuto", "totiž", "trochu",
    "tu", "tuto", "tvůj", "ty", "tyto", "téma", "této", "tě", "těm",
    "těma", "těmu", "u", "už", "v", "ve", "vedle", "vlastně", "však",
    "vy", "vám", "vámi", "vás", "váš", "více", "vsak", "všechen", "všechna",
    "všechno", "všechny", "všichni", "vůbec", "vůči", "z", "za", "zatímco",
    "zde", "ze", "že", "zpět", "zpráva", "zprávy",
}

SLOVAK_STOPWORDS: set[str] = {
    "a", "aby", "aj", "ak", "ako", "ale", "ani", "áno", "asi", "aspoň",
    "bez", "bol", "bola", "boli", "bolo", "by", "byť", "cez", "čo", "dakedy",
    "dnes", "do", "dobre", "dosť", "ho", "hodne", "i", "ja", "jak", "je",
    "jeho", "jej", "jemu", "len", "ma", "majú", "máme", "málo", "mezi", "mi",
    "mňa", "mnou", "moj", "môj", "môže", "my", "na", "nad", "nám", "náš",
    "ne", "nebol", "nebola", "neboli", "nebolo", "nič", "nielen", "od", "o", "on",
    "ona", "oni", "ono", "po", "pod", "podľa", "pokiaľ", "potom", "pre", "pred",
    "pri", "sa", "si", "sme", "som", "ste", "ta", "tak", "takže", "tam", "teda",
    "ten", "tento", "to", "tom", "tomu", "tu", "tvoj", "ty", "u", "v", "vo",
    "vy", "z", "za", "že",
}

ENGLISH_STOPWORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "you", "your", "yours", "yourself", "yourselves",
}

SUPPORTED_LANGUAGE_CODES: set[str] = {"cs", "sk", "en"}
LANGUAGE_ALIASES: dict[str, str] = {
    "czech": "cs",
    "czech language": "cs",
    "slovak": "sk",
    "slovak language": "sk",
    "english": "en",
    "english language": "en",
}
LANGUAGE_STOPWORDS: dict[str, set[str]] = {
    "cs": CZECH_STOPWORDS,
    "sk": SLOVAK_STOPWORDS,
    "en": ENGLISH_STOPWORDS,
}


def normalize_language_code(language: str | None) -> str:
    if not language:
        return "cs"

    normalized = language.strip().lower()
    normalized = LANGUAGE_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_LANGUAGE_CODES:
        raise ValueError(
            f"Unsupported language '{language}'. Use one of: cs, sk, en."
        )
    return normalized


def get_stopwords(language: str | None = None) -> set[str]:
    return set(LANGUAGE_STOPWORDS[normalize_language_code(language)])


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

    def preprocess(self, token: Token, document: str) -> Token | None:
        if token.processed_form in self._stopwords:
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

