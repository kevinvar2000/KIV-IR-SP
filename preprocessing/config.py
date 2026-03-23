from pathlib import Path

try:
    from .tokenizer import TokenType
    from .preprocess import (
        PreprocessingPipeline,
        LowercasePreprocessor,
        LemmatizationPreprocessor,
        MinLengthPreprocessor,
        RemoveDiacriticsPreprocessor,
        RemoveTokenTypesPreprocessor,
        StemmingPreprocessor,
        StopwordPreprocessor,
    )
except ImportError:
    from tokenizer import TokenType
    from preprocess import (
        PreprocessingPipeline,
        LowercasePreprocessor,
        LemmatizationPreprocessor,
        MinLengthPreprocessor,
        RemoveDiacriticsPreprocessor,
        RemoveTokenTypesPreprocessor,
        StemmingPreprocessor,
        StopwordPreprocessor,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "crawler" / "crawled_pages.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "preprocessed"

PIPELINE_NAMES = [
    "baseline",
    "stemming",
    "lemmatization",
    "stemming_no_diacritics",
    "lemmatization_no_diacritics",
]

# Shared base steps used by every pipeline
# These steps are applied to ALL pipelines:
# 1. Lowercase: convert text to lowercase
# 2. RemoveTokenTypes: remove punctuation, HTML tags, and URLs
# 3. StopwordPreprocessor: remove common Czech stopwords
# 4. MinLengthPreprocessor: keep only terms with 2+ characters
# Other pipelines (stemming, lemmatization, etc.) add additional processing on top of these.
BASE_STEPS = [
    LowercasePreprocessor(),
    RemoveTokenTypesPreprocessor({TokenType.PUNCT, TokenType.TAG, TokenType.URL}),
    StopwordPreprocessor(),
    MinLengthPreprocessor(min_length=2),
]


def build_pipelines() -> dict[str, PreprocessingPipeline]:
    return {
        "baseline": PreprocessingPipeline([*BASE_STEPS]),
        "stemming": PreprocessingPipeline([*BASE_STEPS, StemmingPreprocessor()]),
        "lemmatization": PreprocessingPipeline([*BASE_STEPS, LemmatizationPreprocessor()]),
        "stemming_no_diacritics": PreprocessingPipeline([
            *BASE_STEPS,
            StemmingPreprocessor(),
            RemoveDiacriticsPreprocessor(),
        ]),
        "lemmatization_no_diacritics": PreprocessingPipeline([
            *BASE_STEPS,
            LemmatizationPreprocessor(),
            RemoveDiacriticsPreprocessor(),
        ]),
    }
