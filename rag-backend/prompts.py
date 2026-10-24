from config import PROMPT_HISTORY_MESSAGES


def format_context(chunks):
    """Turn retrieved chunks into '[m:ss] text' lines (h:mm:ss after 1 hour)."""
    lines = []
    for c in chunks:
        start = int(c["start"])
        if start >= 3600:
            h = start // 3600
            m = (start % 3600) // 60
            s = start % 60
            ts = f"{h}:{m:02d}:{s:02d}"
        else:
            ts = f"{start // 60}:{start % 60:02d}"
        lines.append(f"[{ts}] {c['text']}")
    return "\n\n".join(lines)


def format_history(history):
    """Last PROMPT_HISTORY_MESSAGES messages as 'role: content' lines."""
    return "\n".join(
        f"{h['role']}: {h['content']}" for h in history[-PROMPT_HISTORY_MESSAGES:]
    ) or "(no previous messages)"


def build_prompt(retrieved, history, query):
    context = format_context(retrieved)
    history_text = format_history(history)
    return f"""You're helping a student who's stuck on a concept while watching a YouTube video.
Use ONLY the transcript context below to answer. If it's not covered, say so honestly.
If the question refers to earlier conversation (like "it"/"that"), use the history to resolve it.

Transcript context:
{context}

Conversation so far:
{history_text}

Student's new question: {query}

Answer clearly and concisely, referencing the timestamp if helpful:"""