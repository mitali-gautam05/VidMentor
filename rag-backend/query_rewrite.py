import logging
import re

from config import (
    REWRITE_ENABLED,
    REWRITE_MAX_CHARS,
    REWRITE_MAX_RETRIES,
    REWRITE_MAX_WORDS,
    REWRITE_REFERENCE_WORDS,
    REWRITE_TIMEOUT_SEC,
)
from llm import call_groq_with_retry
from prompts import build_rewrite_prompt

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def needs_rewrite(query, history):
    """Rewrite only follow-ups: there is history, and the question is short or points back."""
    if not REWRITE_ENABLED or not history:
        return False
    words = _WORD_RE.findall(query.lower())
    if len(words) < REWRITE_MAX_WORDS:
        return True
    return any(w in REWRITE_REFERENCE_WORDS for w in words)


def rewrite_query(query, history):
    """Return a standalone version of `query` for retrieval, or `query` itself.

    The original question is never replaced anywhere else, this text is only used for the
    search. Any failure or odd output (empty, too long) falls back to the raw question.
    """
    if not needs_rewrite(query, history):
        return query

    try:
        rewritten = call_groq_with_retry(
            build_rewrite_prompt(history, query),
            max_retries=REWRITE_MAX_RETRIES,
            timeout=REWRITE_TIMEOUT_SEC,
        )
    except Exception as exc:
        logger.warning("Query rewrite failed, using the original question: %s", exc)
        return query

    rewritten = (rewritten or "").strip().strip('"').strip()
    if not rewritten or len(rewritten) > REWRITE_MAX_CHARS:
        logger.warning("Query rewrite gave an unusable result, using the original question")
        return query
    logger.info("Query rewritten: %r -> %r", query, rewritten)
    return rewritten
