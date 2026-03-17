# Text Preprocessing (Czech IR)

Simple preprocessing pipeline for Czech Information Retrieval experiments.

## What this project does

- Loads documents from `eval_data_cs/documents.json`
- Tokenizes text (words, numbers, dates, URLs, punctuation, tags)
- Runs multiple preprocessing pipelines:
  - baseline
  - stemming
  - lemmatization
  - stemming_no_diacritics
  - lemmatization_no_diacritics
- Builds vocabulary files with term frequencies

## Files

- `main.py` - runs all pipelines and writes output vocabularies
- `tokenizer.py` - tokenization and token types
- `preprocess.py` - preprocessing steps (stopwords, stemming, lemmatization, etc.)

## Requirements

Install dependencies:

```bash
pip install simplemma
```

## How to run

From this folder, run:

```bash
python main.py
```

## Output

The script creates:

- `vocab_baseline.txt`
- `vocab_stemming.txt`
- `vocab_lemmatization.txt`
- `vocab_stemming_no_diacritics.txt`
- `vocab_lemmatization_no_diacritics.txt`
- `vocab.txt` (baseline compatibility output)

Each line in vocabulary files has format:

```text
term frequency
```

## Notes

- The stopword list is Czech-focused.
- Stemming and lemmatization are separate pipelines.
- `eval_data_cs/` also contains query and relevance files for evaluation.
