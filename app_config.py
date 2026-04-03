from pathlib import Path


ROOT = Path(__file__).resolve().parent
CRAWLER_SCRIPT = ROOT / "crawler" / "crawler.py"

CRAWLER_DATA_FILE = ROOT / "data" / "crawler" / "crawled_pages.json"
CRAWLER_LOG_FILE = ROOT / "data" / "crawler" / "crawler.log"
PREPROCESSED_DIR = ROOT / "data" / "preprocessed"
INDEX_DIR = ROOT / "data" / "index"

PIPELINE_OPTIONS = [
    "baseline",
    "stemming",
    "lemmatization",
    "stemming_no_diacritics",
    "lemmatization_no_diacritics",
]
