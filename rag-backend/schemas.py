from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    video_id: str


class ChatRequest(BaseModel):
    video_id: str
    query: str
    current_time: float | None = None
    session_id: str | None = Field(default=None, max_length=64)
    include_sources: bool = False
