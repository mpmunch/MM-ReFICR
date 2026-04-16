import re
from typing import List, Union


def normalize_item_title(text: str) -> str:
    text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("-", " ")
    text = re.sub(r"\([^()]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_variants(title: str) -> List[str]:
    base = normalize_item_title(title)
    no_year = re.sub(r"\s*\(?\d{4}\)?\s*$", "", title).strip()
    no_year_norm = normalize_item_title(no_year)
    variants = [x for x in [base, no_year_norm] if x]
    return list(dict.fromkeys(variants))


def extract_title_from_passage(passage_entry: Union[str, list, tuple]) -> str:
    if isinstance(passage_entry, (tuple, list)):
        if len(passage_entry) == 0:
            return ""
        text = passage_entry[1] if len(passage_entry) > 1 else passage_entry[0]
    else:
        text = passage_entry

    if not isinstance(text, str):
        return ""

    m = re.search(
        r"title\s*:\s*(.*?)(?:\s+actors\s*:|\s+director\s*:|\s+genre\s*:|\s+gneres\s*:|$)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return ""
