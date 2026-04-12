"""Centralized interactive-mode UI messages and menu labels."""

DIVIDER = "=" * 60

# Shared navigation/exit command sets
EXIT_COMMANDS = {"0", "quit", "exit", "q"}
HOME_COMMANDS = {"home", "back"}
NAV_OR_EXIT_COMMANDS = HOME_COMMANDS | EXIT_COMMANDS
APP_EXIT_SIGNAL = "__exit_app__"
APP_EXIT_CODE = 2

# Generic prompts and validation
INVALID_CHOICE = "I didn't understand that option."
INVALID_MENU_OPTION = "I didn't understand that option."
INTERRUPTED_EXIT = "\nInterrupted by user. Exiting interactive mode."
APP_GOODBYE = "\nSession ended."

# Stage runner messages
STAGE_RUNNING = "\n[{name}] running (in-process)"
STAGE_UNHANDLED_EXCEPTION = "[{name}] failed with unhandled exception"
STAGE_FAILED_EXIT_CODE = "[{name}] failed with exit code {exit_code}"
STAGE_FINISHED = "[{name}] finished"

# Shared selection prompts
PROMPT_CHOOSE_NUMBER_NAV = "What would you like to do? (number, 0=exit, home=menu)"
PROMPT_CHOOSE_NAV = "What would you like to do? (number, 0=exit, home=menu)"

# Preprocessing source selection
TITLE_PREPROCESS_SOURCE = "\nChoose source file for preprocessing"
LABEL_TYPE_CUSTOM_FILE = "type custom file path"
PROMPT_ENTER_FILE_TO_PREPROCESS = "Enter file path to preprocess"
ERROR_FILE_NOT_FOUND = "File not found: {path}"

# Language selection
TITLE_LANGUAGE = "Choose document language"
LANGUAGE_OPTIONS = ["czech (cs)", "slovak (sk)", "english (en)"]
LANGUAGE_MAP = {"czech (cs)": "cs", "slovak (sk)": "sk", "english (en)": "en"}

# Pipeline selection
PIPELINE_TITLE = "\nAvailable preprocessing pipelines for {language}:"
PIPELINE_LINE_1 = "1. baseline (lowercase, remove punct/tags/URLs, remove stopwords, min-length 2)"
PIPELINE_LINE_2 = "2. stemming (baseline + language-aware stemming/lemmatization)"
PIPELINE_LINE_3 = "3. lemmatization (baseline + language-aware lemmatization)"
PIPELINE_LINE_4 = "4. stemming_no_diacritics (baseline + stemming + remove accents)"
PIPELINE_LINE_5 = "5. lemmatization_no_diacritics (baseline + lemmatization + remove accents)"
PIPELINE_LINES = [
	PIPELINE_LINE_1,
	PIPELINE_LINE_2,
	PIPELINE_LINE_3,
	PIPELINE_LINE_4,
	PIPELINE_LINE_5,
]
PIPELINE_ALL = "{all_index}. all (runs all pipelines and produces separate output files for each)"
PIPELINE_NOTE = "\nNote: Stopwords, stemming, and lemmatization follow the selected language."
PROMPT_CHOOSE_PIPELINES = "What would you like to do? (numbers comma-separated, 0=exit, home=menu)"
PROMPT_WRITE_VOCAB = "Write vocab files (vocab_<pipeline>.txt)?"

# Text-key detection
ERROR_COULD_NOT_DETECT_TEXT_KEYS = "Could not detect text keys from input file: {path}"
PROMPT_ENTER_TEXT_KEY = "Enter text key manually"
TITLE_DETECTED_TEXT_KEYS = "Detected text keys"

# Crawler
PROMPT_RUN_CRAWLER_BACKGROUND = "Run crawler in background?"

# Preprocessed docs selection / indexing
TITLE_PREPROCESSED_DOCS = "\nChoose preprocessed docs file"
PROMPT_ENTER_PREPROCESSED_DOCS_PATH = "Enter preprocessed docs file path"
INDEXING_EXPECTS_LINE = "\nIndexing expects a preprocessed docs_<pipeline>.jsonl file."
INDEXING_USE_AFTER_PREPROCESS_LINE = "Use this only after preprocessing has already produced normalized documents."

# Retrieval
NO_INDEX_FILES_FOUND = "\nNo index files found under data/index."
PROMPT_BUILD_INDEX_NOW = "Build an index now?"
NO_RETRIEVAL_INDEX = "No retrieval index available."
TITLE_CHOOSE_INDEX_FILE = "Choose index file"

# Top-level interactive menu
INTERACTIVE_MODE_TITLE = "Interactive pipeline mode"
MENU_WHAT_TO_RUN = "\nWhat do you want to run?"
MENU_LINE_1 = "1. crawler"
MENU_LINE_2 = "2. preprocessing + indexing"
MENU_LINE_3 = "3. indexing preprocessed docs"
MENU_LINE_4 = "4. retrieval"
MENU_LINE_EXIT = "0. exit"
MAIN_MENU_LINES = [
	MENU_LINE_1,
	MENU_LINE_2,
	MENU_LINE_3,
	MENU_LINE_4,
	MENU_LINE_EXIT,
]
PROMPT_MAIN_CHOOSE = "What would you like to do? (1..4, 0=exit, home=menu)"
PROMPT_MAIN_DEFAULT = "2"

# Query interface
QUERY_METHOD_TITLE = "SEARCH METHOD"
QUERY_METHOD_LINE_1 = "1. TF-IDF (ranked retrieval with similarity scores)"
QUERY_METHOD_LINE_2 = "2. Boolean (exact term matching with AND/OR/NOT operators)"
QUERY_METHOD_LINE_0 = "0. exit"
PROMPT_QUERY_METHOD = "What would you like to do? (1..2, 0=exit, home=menu): "
QUERY_METHOD_INVALID = "I didn't understand that option. Use 1, 2, 0, home, quit, exit, or q."

QUERY_INTERFACE_TITLE = "QUERY INTERFACE"
QUERY_INTERFACE_HELP = "Enter queries to search. Type home/exit/quit/q/0 to return."
QUERY_INTERFACE_RETURNING = "\nReturning."
QUERY_INTERFACE_SELECTED_METHOD = "Selected method: {method}"
PROMPT_ENTER_QUERY = "\nWhat would you like to search? (home/exit/quit/q/0): "
QUERY_BOOLEAN_FILE_HINT = "Boolean tip: use file:<path> to run one query per line from a text file."
QUERY_FILE_USAGE_HINT = "Use file:<path-to-query-file>."
QUERY_FILE_NOT_FOUND = "Query file not found: {path}"
QUERY_FILE_EMPTY = "No non-empty queries found in file: {path}"
QUERY_FILE_LOADED = "Loaded query file: {path} ({count} queries)"
QUERY_GOODBYE = "\nGoodbye!"
QUERY_INTERRUPTED = "\n\nInterrupted by user."
QUERY_SEARCHING = "\nSearching for: '{query}'"
QUERY_TOTAL_FOUND = "\nTotal documents found: {count}"
QUERY_RESULTS_FOUND = "Results ({count} found):"
QUERY_LOAD_INDEX_ERROR = "Error loading index: {error}"
QUERY_INDEX_NOT_FOUND = "Error: Index file not found: {path}"
QUERY_METADATA_LOAD_NOTE = "Note: Could not load document metadata ({error})"
QUERY_LOADED_INDEX = "\nLoaded index: {name}"
QUERY_FILE_SIZE = "  File size: {size}"
QUERY_PIPELINE = "  Pipeline: {pipeline}"
QUERY_DOCS_TERMS = "  Documents: {docs}, Terms: {terms}"
QUERY_METADATA_COUNT = "  Metadata: {count} documents loaded"
QUERY_PROCESSING_ERROR = "Error processing query: {error}"

QUERY_FORMAT_NO_MATCH = "No matching documents found."
QUERY_FORMAT_HEADER_TFIDF = "│ Rank │ Score │ Title                             │ Link"
QUERY_FORMAT_HEADER_BOOLEAN = "│ Rank   │ Title                             │ Link"
QUERY_FORMAT_MORE_RESULTS = "\n  ... and {count} more results"
