from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import config
import llm
import main
import query_rewrite
from prompts import build_rewrite_prompt
from query_rewrite import needs_rewrite, rewrite_query

HISTORY = [
    {"role": "user", "content": "Explain binary search"},
    {"role": "assistant", "content": "Binary search halves a sorted list each step."},
]


def fail_if_called(*args, **kwargs):
    raise AssertionError("the LLM should not be called")


# ---------- needs_rewrite ----------
def test_no_history_never_rewrites():
    assert needs_rewrite("what's its time complexity?", []) is False


@pytest.mark.parametrize(
    "query",
    [
        "what's its time complexity?",
        "why?",
        "and the second one?",
        "what about the worst case?",
        "iska time complexity?",
    ],
)
def test_short_follow_ups_are_rewritten(query):
    assert needs_rewrite(query, HISTORY) is True


def test_long_standalone_question_is_not_rewritten():
    query = "How does a hash table handle collisions when two keys land in the same bucket?"
    assert needs_rewrite(query, HISTORY) is False


def test_long_question_with_reference_word_is_rewritten():
    query = "Can you walk me through how that works when the list has duplicate values in it?"
    assert needs_rewrite(query, HISTORY) is True


# ---------- rewrite_query ----------
def test_rewrite_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(query_rewrite, "REWRITE_ENABLED", False)
    monkeypatch.setattr(query_rewrite, "call_groq_with_retry", fail_if_called)
    assert rewrite_query("why?", HISTORY) == "why?"


def test_standalone_question_skips_the_llm(monkeypatch):
    monkeypatch.setattr(query_rewrite, "call_groq_with_retry", fail_if_called)
    assert rewrite_query("what's its time complexity?", []) == "what's its time complexity?"


def test_follow_up_uses_the_rewritten_text(monkeypatch):
    monkeypatch.setattr(
        query_rewrite, "call_groq_with_retry", lambda *a, **k: "What is the time complexity of binary search?"
    )
    assert rewrite_query("what's its time complexity?", HISTORY) == "What is the time complexity of binary search?"


def test_rewrite_call_is_cheap_and_gets_history_and_query(monkeypatch):
    seen = {}

    def fake(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["kwargs"] = kwargs
        return "rewritten"

    monkeypatch.setattr(query_rewrite, "call_groq_with_retry", fake)
    rewrite_query("why?", HISTORY)

    assert seen["kwargs"] == {
        "max_retries": config.REWRITE_MAX_RETRIES,
        "timeout": config.REWRITE_TIMEOUT_SEC,
    }
    assert "Binary search halves" in seen["prompt"]
    assert "why?" in seen["prompt"]


def test_output_is_stripped_and_unquoted(monkeypatch):
    monkeypatch.setattr(query_rewrite, "call_groq_with_retry", lambda *a, **k: '  "What is binary search?"\n')
    assert rewrite_query("what is it?", HISTORY) == "What is binary search?"


@pytest.mark.parametrize("bad_output", ["", "   ", None, "x" * (config.REWRITE_MAX_CHARS + 1)])
def test_unusable_output_falls_back_to_the_original(monkeypatch, bad_output):
    monkeypatch.setattr(query_rewrite, "call_groq_with_retry", lambda *a, **k: bad_output)
    assert rewrite_query("why?", HISTORY) == "why?"


def test_llm_failure_falls_back_to_the_original(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("groq is down")

    monkeypatch.setattr(query_rewrite, "call_groq_with_retry", boom)
    assert rewrite_query("why?", HISTORY) == "why?"


def test_rewrite_prompt_contains_history_and_question():
    prompt = build_rewrite_prompt(HISTORY, "what's its time complexity?")
    assert "Explain binary search" in prompt
    assert "what's its time complexity?" in prompt


# ---------- llm timeout option ----------
def test_call_groq_uses_the_per_call_timeout(monkeypatch):
    response = MagicMock()
    response.choices[0].message.content = "ok"
    scoped = MagicMock()
    scoped.chat.completions.create.return_value = response
    fake_client = MagicMock()
    fake_client.with_options.return_value = scoped
    monkeypatch.setattr(llm, "groq_client", fake_client)

    assert llm.call_groq_with_retry("hi", max_retries=1, timeout=5.0) == "ok"
    fake_client.with_options.assert_called_once_with(timeout=5.0)


# ---------- /chat wiring ----------
@pytest.fixture
def clean_chat_sessions():
    main.chat_sessions.clear()
    yield
    main.chat_sessions.clear()


def test_chat_searches_with_the_rewrite_but_keeps_the_original(monkeypatch, clean_chat_sessions):
    chunks = [{"text": "Binary search runs in O(log n).", "start": 12, "end": 52}]
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: ((chunks, object()), None))
    main.chat_sessions[("vid_rw", "vid_rw")] = list(HISTORY)

    searched = {}

    def fake_retrieve(query, chunks, idx, ct):
        searched["query"] = query
        return chunks

    prompts = []
    monkeypatch.setattr(main, "retrieve", fake_retrieve)
    monkeypatch.setattr(main, "rewrite_query", lambda q, h: "What is the time complexity of binary search?")
    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: prompts.append(prompt) or "O(log n)")

    response = TestClient(main.app).post(
        "/chat", json={"video_id": "vid_rw", "query": "what's its time complexity?", "current_time": 20.0}
    )

    assert response.status_code == 200
    assert searched["query"] == "What is the time complexity of binary search?"
    assert "Student's new question: what's its time complexity?" in prompts[0]
    assert main.chat_sessions[("vid_rw", "vid_rw")][-2]["content"] == "what's its time complexity?"
