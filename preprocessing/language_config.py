"""Language resources for preprocessing."""

# Czech stopwords - standard list used in Czech NLP / IR
CZECH_STOPWORDS: set[str] = {
    "a", "aby", "aj", "ale", "ani", "aniz", "ano", "asi", "aspon", "atd",
    "atp", "az", "ackoli", "az", "bez", "beze", "blizko", "bohuzel", "brzo",
    "brzy", "bud", "budu", "by", "byl", "byla", "byli", "bylo", "byly",
    "bys", "byt", "behem", "co", "coz", "coz", "cz", "ci", "clanek",
    "clanku", "clanky", "dalsi", "dnes", "do", "dokonce", "dokud", "dost",
    "dosud", "doufam", "dva", "dve", "dal", "dale", "dekovat", "dekuji",
    "ho", "hodne", "i", "jak", "jakmile", "jako", "jakoz", "jaky", "je",
    "jeden", "jedna", "jednak", "jedno", "jednou", "jedny", "jeho", "jej",
    "jeji", "jejich", "jemu", "jen", "jenz", "jestli", "jestlize", "jeste",
    "jez", "ji", "jich", "jimi", "jinak", "jine", "jini", "jiny", "jiz",
    "jsi", "jsme", "jsou", "jste", "ja", "ji", "jim", "k", "kam", "kde",
    "kdo", "kdy", "kdyz", "ke", "kolem", "kolik", "krome", "kratce", "kratky",
    "ktera", "ktere", "kteri", "ktery", "kvuli", "ma", "maji", "malo", "me",
    "mezi", "mi", "mne", "mnou", "mne", "moc", "mohl", "mohou", "moje",
    "moji", "mozna", "muj", "musi", "my", "myslim", "ma", "mam", "mate",
    "mit", "me", "muj", "muze", "na", "nad", "nade", "nam", "naproti",
    "nas", "nase", "nasi", "ne", "nebot", "nebo", "nebyl", "nebyla", "nebyli",
    "nebylo", "nebyly", "necht", "nedela", "nedelaji", "nedelam", "nedelate",
    "neg", "nej", "nejsou", "nekde", "nekdo", "nektera", "nektere", "nektery",
    "nemaji", "nema", "nemel", "nemuze", "nes", "nesmi", "nesmi", "nez",
    "nic", "nichz", "ni", "nim", "nimi", "no", "nove", "novy", "nas",
    "nam", "na", "neco", "nejak", "nejaky", "nemu", "nemuz", "o", "od",
    "ode", "on", "ona", "oni", "ono", "ony", "ostatne", "pak", "pan", "pani",
    "po", "pod", "podle", "pokud", "pouze", "potom", "pote", "porad",
    "prav", "prave", "pro", "proc", "proste", "prosim", "proto", "protoze",
    "prvni", "pred", "prede", "pres", "prese", "presto", "pri", "pricemz",
    "re", "rovnez", "s", "sam", "se", "si", "sice", "skoro", "smi", "smeji",
    "snad", "spolu", "sta", "strana", "sve", "sveho", "svou", "svuj", "svym",
    "svymi", "sa", "sam", "sve", "ta", "tak", "take", "takze", "tam",
    "tamhle", "tamhleto", "tamto", "tato", "tedy", "ten", "tento", "ti",
    "tim", "timto", "to", "toho", "tohle", "tomu", "tomuto", "totiz", "trochu",
    "tu", "tuto", "tvuj", "ty", "tyto", "tema", "teto", "te", "tem",
    "tema", "temu", "u", "uz", "v", "ve", "vedle", "vlastne", "vsak",
    "vy", "vam", "vami", "vas", "vas", "vice", "vsak", "vsechen", "vsechna",
    "vsechno", "vsechny", "vsichni", "vubec", "vuci", "z", "za", "zatimco",
    "zde", "ze", "ze", "zpet", "zprava", "zpravy",
}

SLOVAK_STOPWORDS: set[str] = {
    "a", "aby", "aj", "ak", "ako", "ale", "ani", "ano", "asi", "aspon",
    "bez", "bol", "bola", "boli", "bolo", "by", "byt", "cez", "co", "dakedy",
    "dnes", "do", "dobre", "dost", "ho", "hodne", "i", "ja", "jak", "je",
    "jeho", "jej", "jemu", "len", "ma", "maju", "mame", "malo", "mezi", "mi",
    "mna", "mnou", "moj", "moj", "moze", "my", "na", "nad", "nam", "nas",
    "ne", "nebol", "nebola", "neboli", "nebolo", "nic", "nielen", "od", "o", "on",
    "ona", "oni", "ono", "po", "pod", "podla", "pokial", "potom", "pre", "pred",
    "pri", "sa", "si", "sme", "som", "ste", "ta", "tak", "takze", "tam", "teda",
    "ten", "tento", "to", "tom", "tomu", "tu", "tvoj", "ty", "u", "v", "vo",
    "vy", "z", "za", "ze",
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