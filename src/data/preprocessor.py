"""
Text preprocessing and transcript section splitting.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Patterns that mark the start of the Q&A section
_QA_PATTERNS = [
    r"question[- ]and[- ]answer",
    r"q&a session",
    r"q&a",
    r"open.*floor.*question",
    r"operator.*question",
    r"we.*now.*open.*question",
]
_QA_RE = re.compile("|".join(_QA_PATTERNS), re.IGNORECASE)


def clean_text(text: str) -> str:
    """Basic cleaning: collapse whitespace, strip boilerplate."""
    if not isinstance(text, str):
        return ""
    # Remove page-break artifacts and excess whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip operator/moderator cues that add noise
    text = re.sub(r"\b(operator|moderator):\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def split_sections(text: str) -> dict:
    """
    Split a transcript into:
        "prepared"  – everything before the Q&A section
        "qa"        – the Q&A section itself
        "full"      – the complete (cleaned) transcript

    If no Q&A marker is found, "prepared" == "full" and "qa" == "".
    """
    text = clean_text(text)
    match = _QA_RE.search(text)
    if match:
        prepared = text[:match.start()].strip()
        qa       = text[match.start():].strip()
    else:
        prepared = text
        qa       = ""
    return {"full": text, "prepared": prepared, "qa": qa}
