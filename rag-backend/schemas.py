from pydantic import BaseModel, Field
from typing import Annotated

VideoId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{11}$")]


class IndexRequest(BaseModel):
    video_id: VideoId


class ChatRequest(BaseModel):
    video_id: VideoId
    query: str
    current_time: float | None = None
    session_id: str | None = Field(default=None, max_length=64)
