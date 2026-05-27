import re

from domains._toon_io import encode_toon


CORE_SECTION_KEYWORDS = [
    "abstract",
    "introduction",
    "background",
    "related",
    "method",
    "approach",
    "model",
    "algorithm",
    "experiment",
    "evaluation",
    "result",
    "analysis",
    "discussion",
    "limitation",
    "conclusion",
]
DROP_SECTION_KEYWORDS = [
    "reference",
    "bibliography",
    "appendix",
    "acknowledg",
    "supplement",
]
DROP_MARKERS = ["\nFigures:", "\nFormulas:"]
KEY_TABLE_KEYWORDS = [
    "ablation",
    "accuracy",
    "auc",
    "baseline",
    "benchmark",
    "bleu",
    "comparison",
    "evaluation",
    "experiment",
    "f1",
    "performance",
    "precision",
    "recall",
    "result",
    "score",
]
RETRIEVAL_KEYWORDS = [
    "ablation",
    "accuracy",
    "baseline",
    "benchmark",
    "claim",
    "contribution",
    "evaluation",
    "experiment",
    "improve",
    "limitation",
    "method",
    "novel",
    "outperform",
    "performance",
    "result",
    "significant",
    "state-of-the-art",
]
RETRIEVAL_SECTION_WEIGHTS = {
    "result": 8,
    "evaluation": 8,
    "experiment": 7,
    "method": 6,
    "approach": 6,
    "model": 5,
    "algorithm": 5,
    "limitation": 4,
    "discussion": 3,
    "conclusion": 3,
    "introduction": 2,
}
DROP_TABLE_KEYWORDS = [
    "configuration",
    "dataset statistics",
    "hyperparameter",
    "implementation detail",
    "parameter sweep",
    "sweep",
]


def normalize_text(text):
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate(text, char_limit):
    text = normalize_text(text)
    if len(text) <= char_limit:
        return text
    return text[:char_limit].rstrip() + "\n[truncated]"


def _drop_back_matter(text):
    cut = len(text)
    for marker in DROP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def _extract_figures_text(text):
    match = re.search(
        r"(?ms)(?:^|\n)Figures:\s*(.*?)(?=\nFormulas:|\Z)",
        text,
    )
    return match.group(1) if match else ""


def _figure_blocks(figures_text):
    return [
        match.group(0)
        for match in re.finditer(
            r"(?ms)^Figure\s+.*?(?=^Figure\s+|\Z)",
            figures_text,
        )
    ]


def _field_value(block, field_name):
    match = re.search(rf"(?mi)^{re.escape(field_name)}:\s*(.*)$", block)
    return normalize_text(match.group(1)) if match else ""


def _data_value(block):
    match = re.search(r"(?mis)^Data:\s*(.*)$", block)
    return normalize_text(match.group(1)) if match else ""


def _is_key_table(caption, data):
    content = f"{caption} {data}".lower()
    if any(keyword in content for keyword in DROP_TABLE_KEYWORDS):
        return False
    return any(keyword in content for keyword in KEY_TABLE_KEYWORDS)


def _extract_key_tables(text, table_char_limit, max_key_tables):
    key_tables = []
    for block in _figure_blocks(_extract_figures_text(text)):
        if _field_value(block, "Type").lower() != "table":
            continue

        caption = _field_value(block, "Caption")
        data = _data_value(block)
        if not caption or not data:
            continue
        if not _is_key_table(caption, data):
            continue

        key_tables.append(
            {
                "caption": _truncate(caption, table_char_limit),
                "data": _truncate(data, table_char_limit),
            }
        )
        if len(key_tables) >= max_key_tables:
            break
    return key_tables


def _extract_title(text):
    match = re.search(r"^\s*['\"]?Title:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    return normalize_text(match.group(1)) if match else ""


def _split_sections(text):
    matches = list(re.finditer(r"(?m)^Section:\s*(.+?)\s*$", text))
    sections = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        name = normalize_text(match.group(1))
        content = normalize_text(text[start:end])
        sections.append({"name": name, "content": content})
    return sections


def _extract_abstract(text, sections):
    for section in sections:
        if section["name"].lower() == "abstract":
            return section["content"]
    match = re.search(
        r"Abstract:\s*(.*?)(?=\n\s*Section:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return normalize_text(match.group(1)) if match else ""


def _should_drop(name):
    lower_name = name.lower()
    return any(keyword in lower_name for keyword in DROP_SECTION_KEYWORDS)


def _is_core(name):
    lower_name = name.lower()
    return any(keyword in lower_name for keyword in CORE_SECTION_KEYWORDS)


def _selected_core_sections(sections, section_char_limit, max_sections):
    selected_sections = []
    dropped_sections = []
    for section in sections:
        name = section["name"]
        if name.lower() == "abstract":
            continue
        if _should_drop(name):
            dropped_sections.append(name)
            continue
        if _is_core(name):
            selected_sections.append(
                {
                    "name": name,
                    "content": _truncate(section["content"], section_char_limit),
                }
            )
        else:
            dropped_sections.append(name)
        if len(selected_sections) >= max_sections:
            break
    return selected_sections, dropped_sections


def _section_weight(name):
    lower_name = name.lower()
    return sum(
        weight
        for keyword, weight in RETRIEVAL_SECTION_WEIGHTS.items()
        if keyword in lower_name
    )


def _retrieval_score(section):
    name = section["name"]
    content = section["content"].lower()
    score = _section_weight(name)
    score += sum(content.count(keyword) for keyword in RETRIEVAL_KEYWORDS)
    return score


def _core_text(title, abstract, selected_sections):
    blocks = []
    if title:
        blocks.append(f"Title: {title}")
    if abstract:
        blocks.append(f"Abstract:\n{abstract}")
    for section in selected_sections:
        blocks.append(f"Section: {section['name']}\n{section['content']}")
    return "\n\n".join(blocks)


def build_review_packet(
    paper_text,
    section_char_limit=5000,
    abstract_char_limit=2200,
    max_sections=10,
    table_char_limit=2400,
    max_key_tables=4,
):
    full_text = normalize_text(paper_text)
    key_tables = _extract_key_tables(full_text, table_char_limit, max_key_tables)
    text = _drop_back_matter(full_text)
    title = _extract_title(text)
    sections = _split_sections(text)
    abstract = _extract_abstract(text, sections)
    selected_sections, dropped_sections = _selected_core_sections(
        sections,
        section_char_limit,
        max_sections,
    )

    return {
        "title": title,
        "abstract": _truncate(abstract, abstract_char_limit),
        "sections": selected_sections,
        "key_tables": key_tables,
        "dropped_sections": dropped_sections,
        "packet_notes": [
            "Compact review packet derived from raw paper_text.",
            "References, appendices, non-key figure dumps, formula dumps, and non-core sections are omitted or truncated.",
        ],
    }


def build_review_packet_toon(paper_text, **kwargs):
    return encode_toon(build_review_packet(paper_text, **kwargs))


def build_core_text_packet_parts(
    paper_text,
    section_char_limit=10000,
    abstract_char_limit=3000,
    max_sections=12,
    table_char_limit=2400,
    max_key_tables=4,
):
    full_text = normalize_text(paper_text)
    key_tables = _extract_key_tables(full_text, table_char_limit, max_key_tables)
    text = _drop_back_matter(full_text)
    title = _extract_title(text)
    sections = _split_sections(text)
    abstract = _truncate(_extract_abstract(text, sections), abstract_char_limit)
    selected_sections, dropped_sections = _selected_core_sections(
        sections,
        section_char_limit,
        max_sections,
    )
    metadata = {
        "title": title,
        "included_sections": [section["name"] for section in selected_sections],
        "key_tables": key_tables,
        "dropped_sections": dropped_sections,
        "packet_notes": [
            "Raw core text preserves selected paper prose.",
            "TOON sidecar carries title, included section names, key tables, and dropped section names.",
            "References, appendices, non-key figure dumps, and formula dumps are omitted.",
        ],
    }
    return {
        "metadata_toon": encode_toon(metadata),
        "core_text": _core_text(title, abstract, selected_sections),
    }


def build_retrieval_packet(
    paper_text,
    chunk_char_limit=3200,
    abstract_char_limit=2200,
    max_chunks=8,
    table_char_limit=2400,
    max_key_tables=4,
):
    full_text = normalize_text(paper_text)
    key_tables = _extract_key_tables(full_text, table_char_limit, max_key_tables)
    text = _drop_back_matter(full_text)
    title = _extract_title(text)
    sections = _split_sections(text)
    abstract = _truncate(_extract_abstract(text, sections), abstract_char_limit)

    candidates = [
        section
        for section in sections
        if section["name"].lower() != "abstract"
        and not _should_drop(section["name"])
        and _is_core(section["name"])
    ]
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (_retrieval_score(item[1]), -item[0]),
        reverse=True,
    )

    evidence_chunks = []
    for rank, (source_index, section) in enumerate(ranked[:max_chunks], start=1):
        evidence_chunks.append(
            {
                "chunk_id": f"c{rank}",
                "section": section["name"],
                "source_order": source_index + 1,
                "content": _truncate(section["content"], chunk_char_limit),
            }
        )

    return {
        "title": title,
        "abstract": abstract,
        "evidence_chunks": evidence_chunks,
        "key_tables": key_tables,
        "packet_notes": [
            "Retrieval-style packet ranked by review-relevant section names and evidence keywords.",
            "References, appendices, non-key figure dumps, formula dumps, and lower-scoring sections are omitted.",
        ],
    }


def build_retrieval_packet_toon(paper_text, **kwargs):
    return encode_toon(build_retrieval_packet(paper_text, **kwargs))
