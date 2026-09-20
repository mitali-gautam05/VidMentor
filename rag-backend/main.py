from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from sentence_transformers import SentenceTransformer
import faiss, os, pickle, time
from groq import Groq



app = FastAPI()

# Chrome extensions call from "chrome-extension://<id>" origin — allow that
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your extension's exact origin once you have the ID
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))  # set as env var, never hardcode

CACHE_DIR = "rag_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
video_cache = {}
chat_sessions = {}

# ---------- Transcript + chunking ----------
def get_transcript(video_id, preferred_langs=("en", "hi")):
    ytt_api = YouTubeTranscriptApi()
    try:
        fetched = ytt_api.fetch(video_id, languages=list(preferred_langs))
        return fetched.to_raw_data(), None
    except NoTranscriptFound:
        try:
            transcript_list = ytt_api.list(video_id)
            first_available = next(iter(transcript_list))
            return first_available.fetch().to_raw_data(), None
        except Exception:
            return None, "No transcript available in any language for this video."
    except TranscriptsDisabled:
        return None, "Captions are disabled for this video."
    except VideoUnavailable:
        return None, "Video not found or unavailable."
    except Exception as e:
        return None, f"Unexpected error: {e}"

def chunk_transcript(transcript, window_sec=40):
    chunks = []
    current_text, current_start = [], transcript[0]['start']
    for seg in transcript:
        if seg['start'] - current_start > window_sec and current_text:
            chunks.append({"text": " ".join(current_text), "start": current_start, "end": seg['start']})
            current_text, current_start = [], seg['start']
        current_text.append(seg['text'])
    if current_text:
        chunks.append({"text": " ".join(current_text), "start": current_start, "end": transcript[-1]['start']})
    return chunks

# ---------- Index build + cache ----------
def build_index_for_video(video_id):
    transcript, error = get_transcript(video_id)
    if transcript is None:
        return None, error
    chunks = chunk_transcript(transcript)
    embeddings = embedder.encode([c["text"] for c in chunks], convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return (chunks, index), None

def save_to_disk(video_id, chunks, index):
    faiss.write_index(index, f"{CACHE_DIR}/{video_id}.index")
    with open(f"{CACHE_DIR}/{video_id}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_from_disk(video_id):
    idx_path, chunks_path = f"{CACHE_DIR}/{video_id}.index", f"{CACHE_DIR}/{video_id}_chunks.pkl"
    if os.path.exists(idx_path) and os.path.exists(chunks_path):
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        return chunks, faiss.read_index(idx_path)
    return None

def get_or_build_index(video_id):
    if video_id in video_cache:
        return video_cache[video_id], None
    disk_result = load_from_disk(video_id)
    if disk_result:
        video_cache[video_id] = disk_result
        return disk_result, None
    result, error = build_index_for_video(video_id)
    if result is None:
        return None, error
    chunks, index = result
    save_to_disk(video_id, chunks, index)
    video_cache[video_id] = (chunks, index)
    return (chunks, index), None

# ---------- Retrieval ----------
# fixed time-window chunking strategy like timestamp based
def retrieve(query, chunks, index, current_time=None, top_k=4, time_window=180):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_emb, len(chunks))
    results = []
    for score, i in zip(scores[0], idxs[0]):
        c = chunks[i]
        penalty = 0
        if current_time is not None:
            dist = abs(c["start"] - current_time)
            penalty = max(0, dist - time_window) * 0.001
        results.append((score - penalty, c))
    results.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in results[:top_k]]

def call_groq_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"Groq API error: {e}"

# ---------- API models ----------
class IndexRequest(BaseModel):
    video_id: str

class ChatRequest(BaseModel):
    video_id: str
    query: str
    current_time: float | None = None

# ---------- Endpoints ----------
@app.post("/index")
def index_video(req: IndexRequest):
    result, error = get_or_build_index(req.video_id)
    if result is None:
        return {"success": False, "error": error}
    return {"success": True, "message": "Video indexed and ready."}

@app.post("/chat")
def chat(req: ChatRequest):
    result, error = get_or_build_index(req.video_id)
    if result is None:
        return {"success": False, "error": error}
    chunks, index = result

    retrieved = retrieve(req.query, chunks, index, req.current_time)
    context = "\n\n".join(
        f"[{int(c['start']//60)}:{int(c['start']%60):02d}] {c['text']}" for c in retrieved
    )
    history = chat_sessions.get(req.video_id, [])
    history_text = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:]) or "(no previous messages)"

    prompt = f"""You're helping a student who's stuck on a concept while watching a YouTube video.
Use ONLY the transcript context below to answer. If it's not covered, say so honestly.
If the question refers to earlier conversation (like "it"/"that"), use the history to resolve it.

Transcript context:
{context}

Conversation so far:
{history_text}

Student's new question: {req.query}

Answer clearly and concisely, referencing the timestamp if helpful:"""

    answer = call_groq_with_retry(prompt)
    history.append({"role": "user", "content": req.query})
    history.append({"role": "assistant", "content": answer})
    chat_sessions[req.video_id] = history

    return {"success": True, "answer": answer}

@app.get("/health")
def health():
    return {"status": "ok"}