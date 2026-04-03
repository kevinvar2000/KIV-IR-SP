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
        normalize_language_code,
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
        normalize_language_code,
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

def build_base_steps(language: str = "cs") -> list:
    normalized_language = normalize_language_code(language)
    return [
        LowercasePreprocessor(),
        RemoveTokenTypesPreprocessor({TokenType.PUNCT, TokenType.TAG, TokenType.URL}),
        StopwordPreprocessor(language=normalized_language),
        MinLengthPreprocessor(min_length=2),
    ]


def build_pipelines(language: str = "cs") -> dict[str, PreprocessingPipeline]:
    normalized_language = normalize_language_code(language)
    base_steps = build_base_steps(normalized_language)
    return {
        "baseline": PreprocessingPipeline([*base_steps]),
        "stemming": PreprocessingPipeline([*base_steps, StemmingPreprocessor(normalized_language)]),
        "lemmatization": PreprocessingPipeline([*base_steps, LemmatizationPreprocessor(normalized_language)]),
        "stemming_no_diacritics": PreprocessingPipeline([
            *base_steps,
            StemmingPreprocessor(normalized_language),
            RemoveDiacriticsPreprocessor(),
        ]),
        "lemmatization_no_diacritics": PreprocessingPipeline([
            *base_steps,
            LemmatizationPreprocessor(normalized_language),
            RemoveDiacriticsPreprocessor(),
        ]),
    }
