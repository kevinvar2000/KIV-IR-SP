from typing import Dict, List

from .tfidf import InvertedIndex


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    """Return an ASCII table for pretty terminal output."""

    table_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]

    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"

    lines = [separator, header_line, separator]
    for row in table_rows:
        line = "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        lines.append(line)

    lines.append(separator)

    return "\n".join(lines)


def build_term_breakdown_rows(
    term_freqs: Dict[str, int],
    index: InvertedIndex,
    vector: Dict[str, float],
    norm: float,
) -> List[List[str]]:
    """Create per-term rows used by the printed TF-IDF tables."""

    rows: List[List[str]] = []
    for term in sorted(term_freqs.keys()):
        tf = term_freqs[term]
        wtf = index.weighted_tf(tf)
        tf_idf = vector.get(term, 0.0)

        normalized = tf_idf / norm if norm != 0 else 0.0
        rows.append([
            term,
            str(tf),
            f"{wtf:.6f}",
            f"{tf_idf:.6f}",
            f"{normalized:.6f}",
        ])

    return rows
