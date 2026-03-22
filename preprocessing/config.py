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
