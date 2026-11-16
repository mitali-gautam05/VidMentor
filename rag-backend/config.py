import os
from dotenv import load_dotenv

load_dotenv()

# Secrets (from .env, never hardcode)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Models
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0.3

# Transcript
PREFERRED_LANGS = ("en", "hi")

# Chunking
WINDOW_SEC = 40
# Seconds each chunk repeats from the end of the previous one (0 = no overlap).
OVERLAP_SEC = int(os.environ.get("OVERLAP_SEC", "0"))

# Retrieval
TOP_K = 4
TIME_WINDOW_SEC = 180
TIME_PENALTY_PER_SEC = 0.001

# LLM calls and chat history
# Timeout in seconds for individual Groq API requests (per attempt). Default of 30.0s accommodates
# model generation latency on Groq while bounding wait time for the interactive Chrome extension.
GROQ_TIMEOUT_SEC = float(os.environ.get("GROQ_TIMEOUT_SEC", 30.0))
# Total attempts allowed for LLM calls (1 initial attempt + up to 2 retries)
MAX_RETRIES = 3
# Maximum delay between retry attempts in seconds to prevent unbounded backoff
MAX_RETRY_DELAY_SEC = 8.0
# Messages of recent conversation fed into the prompt (kept separate from the
# in-process memory cap below so prompt length and memory bounds stay independent)
PROMPT_HISTORY_MESSAGES = 6

# In-process chat memory: cap the messages kept per session and the sessions kept at all
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "20"))
# Force even so whole exchanges are never split in half.
MAX_HISTORY_MESSAGES = (MAX_HISTORY_MESSAGES // 2) * 2
MAX_CHAT_SESSIONS = int(os.environ.get("MAX_CHAT_SESSIONS", "100"))

# Query rewriting for follow-up questions (see query_rewrite.py).
# With no chat history the rewrite is skipped. Otherwise it runs when the question is short
# or contains a reference word. The LLM call is kept cheap: one attempt, short timeout.
# Set REWRITE_ENABLED=0 to skip the rewrite (every follow-up costs one extra Groq call).
REWRITE_ENABLED = os.environ.get("REWRITE_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
REWRITE_MAX_WORDS = 8
REWRITE_MAX_CHARS = 200
REWRITE_MAX_RETRIES = 1
REWRITE_TIMEOUT_SEC = float(os.environ.get("REWRITE_TIMEOUT_SEC", 8.0))
REWRITE_REFERENCE_WORDS = (
    "it", "its", "this", "that", "these", "those", "they", "them", "he", "she", "him", "her",
    "above", "previous", "earlier", "again", "former", "latter", "second", "third",
    # Hinglish
    "iska", "iski", "iske", "uska", "uski", "uske", "isme", "usme", "yeh", "ye", "woh", "wo",
)

# Cache
# "disk" (default) or "redis". Redis is only imported when it is selected.
CACHE_BACKEND = os.environ.get("CACHE_BACKEND", "disk").strip().lower()
# Folder for the disk backend. Point it at a persistent volume (e.g. /data/cache) to keep indexes.
CACHE_DIR = os.environ.get("CACHE_DIR", "rag_cache")
# Only used when CACHE_BACKEND=redis.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
# Seconds before a Redis entry expires. Unset, 0 or invalid means no expiry.
try:
    CACHE_TTL_SEC = int(os.environ.get("CACHE_TTL_SEC", "0"))
except ValueError:
    CACHE_TTL_SEC = 0
CACHE_TTL_SEC = max(CACHE_TTL_SEC, 0)
