from pydantic import BaseModel, Field
from typing import Annotated

VideoId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{11}$")]


class TranscriptSegment(BaseModel):
    text: str = Field(..., max_length=5000)
    start: float = Field(..., ge=0.0)
    duration: float = Field(default=0.0, ge=0.0)


class IndexRequest(BaseModel):
    video_id: VideoId
    transcript: list[TranscriptSegment] | None = Field(default=None, max_length=5000)


class ChatRequest(BaseModel):
    video_id: VideoId
    query: str
    current_time: float | None = None
    session_id: str | None = Field(default=None, max_length=64)
    transcript: list[TranscriptSegment] | None = Field(default=None, max_length=5000)
