import re
import logging

logger = logging.getLogger(__name__)

_QA_RE = re.compile(
    "|".join([
        r"question[- ]and[- ]answer",
        r"q&a session",
        r"q&a",
        r"open.*floor.*question",
        r"operator.*question",
        r"we.*now.*open.*question",
    ]),
    re.IGNORECASE,
)

_HANDOFF_RE = re.compile(
    r"(?:turn(?:ing)?\s+(?:the\s+)?(?:call|floor|conference|meeting|time)\s+over\s+to|"
    r"hand(?:ing)?\s+(?:the\s+)?(?:call|floor)\s+(?:over\s+)?to|"
    r"introduce\s+(?:our|your))"
    r"[^.\n]{0,200}\.",
    re.IGNORECASE,
)

_QA_PASS_RE = re.compile(
    r"Operator:\s*"
    r"(?:Thank\s+you\.?\s*)?"
    r"(?:Our\s+(?:next|first|second|third|fourth|fifth|last)\s+question\s+(?:comes\s+|is\s+)?from|"
    r"The\s+(?:next|last)\s+question\s+(?:comes\s+|is\s+)?from|"
    r"Your\s+next\s+question\s+(?:comes\s+|is\s+)?from|"
    r"We\s+have\s+(?:a\s+question|our\s+next\s+question)\s+from)"
    r"[^.\n]*?\.\s*",
    re.IGNORECASE,
)

_CLOSING_RE = re.compile(
    r"Operator:[^\n]{0,200}?(?:no\s+further\s+questions|"
    r"conclude(?:s)?\s+(?:today's\s+|this\s+|our\s+)?(?:teleconference|conference\s+call|call|presentation)|"
    r"thank\s+you\s+for\s+(?:your\s+)?participat(?:ion|ing)|"
    r"you\s+may\s+(?:now\s+)?disconnect)",
    re.IGNORECASE,
)

_DISCLAIMER_RE = re.compile(
    r"[^.\n]*?(?:forward[- ]looking\s+statements?|"
    r"safe\s+harbor\s+provisions?|"
    r"actual\s+results\s+may\s+differ\s+materially)"
    r"[^.\n]*?\.",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(operator|moderator):\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def strip_boilerplate(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    original = text

    m = _HANDOFF_RE.search(text)
    if m:
        text = text[m.end():]

    text = _QA_PASS_RE.sub("", text)

    m = _CLOSING_RE.search(text)
    if m:
        text = text[:m.start()]

    text = _DISCLAIMER_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 0.30 * len(original):
        text = re.sub(r"\s+", " ", original).strip()
    return text


def split_sections(text: str) -> dict:
    text = clean_text(text)
    match = _QA_RE.search(text)
    if match:
        prepared = text[:match.start()].strip()
        qa = text[match.start():].strip()
    else:
        prepared = text
        qa = ""
    return {"full": text, "prepared": prepared, "qa": qa}
