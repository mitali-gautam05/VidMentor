import pytest
from pydantic import ValidationError

from schemas import IndexRequest, ChatRequest


class TestIndexRequestVideoId:
    def test_valid_video_id(self):
        req = IndexRequest(video_id="dQw4w9WgXcQ")
        assert req.video_id == "dQw4w9WgXcQ"

    def test_valid_video_id_with_underscore(self):
        req = IndexRequest(video_id="abc_def_ghi")
        assert req.video_id == "abc_def_ghi"

    def test_valid_video_id_with_hyphen(self):
        req = IndexRequest(video_id="abc-def-ghi")
        assert req.video_id == "abc-def-ghi"

    def test_too_short_video_id(self):
        with pytest.raises(ValidationError) as exc:
            IndexRequest(video_id="short")
        assert "video_id" in str(exc.value)

    def test_too_long_video_id(self):
        with pytest.raises(ValidationError) as exc:
            IndexRequest(video_id="dQw4w9WgXcQextra")
        assert "video_id" in str(exc.value)

    def test_path_traversal_video_id(self):
        with pytest.raises(ValidationError) as exc:
            IndexRequest(video_id="../x")
        assert "video_id" in str(exc.value)

    def test_slash_in_video_id(self):
        with pytest.raises(ValidationError) as exc:
            IndexRequest(video_id="abc/def")
        assert "video_id" in str(exc.value)

    def test_space_in_video_id(self):
        with pytest.raises(ValidationError) as exc:
            IndexRequest(video_id="abc def ghi")
        assert "video_id" in str(exc.value)

    def test_empty_video_id(self):
        with pytest.raises(ValidationError) as exc:
            IndexRequest(video_id="")
        assert "video_id" in str(exc.value)

    def test_special_chars_in_video_id(self):
        with pytest.raises(ValidationError) as exc:
            IndexRequest(video_id="abc@def#ghi")
        assert "video_id" in str(exc.value)


class TestChatRequestVideoId:
    def test_valid_video_id(self):
        req = ChatRequest(video_id="dQw4w9WgXcQ", query="test")
        assert req.video_id == "dQw4w9WgXcQ"

    def test_valid_video_id_with_underscore(self):
        req = ChatRequest(video_id="abc_def_ghi", query="test")
        assert req.video_id == "abc_def_ghi"

    def test_valid_video_id_with_hyphen(self):
        req = ChatRequest(video_id="abc-def-ghi", query="test")
        assert req.video_id == "abc-def-ghi"

    def test_too_short_video_id(self):
        with pytest.raises(ValidationError) as exc:
            ChatRequest(video_id="short", query="test")
        assert "video_id" in str(exc.value)

    def test_too_long_video_id(self):
        with pytest.raises(ValidationError) as exc:
            ChatRequest(video_id="dQw4w9WgXcQextra", query="test")
        assert "video_id" in str(exc.value)

    def test_path_traversal_video_id(self):
        with pytest.raises(ValidationError) as exc:
            ChatRequest(video_id="../x", query="test")
        assert "video_id" in str(exc.value)

    def test_slash_in_video_id(self):
        with pytest.raises(ValidationError) as exc:
            ChatRequest(video_id="abc/def", query="test")
        assert "video_id" in str(exc.value)

    def test_space_in_video_id(self):
        with pytest.raises(ValidationError) as exc:
            ChatRequest(video_id="abc def ghi", query="test")
        assert "video_id" in str(exc.value)

    def test_empty_video_id(self):
        with pytest.raises(ValidationError) as exc:
            ChatRequest(video_id="", query="test")
        assert "video_id" in str(exc.value)

    def test_special_chars_in_video_id(self):
        with pytest.raises(ValidationError) as exc:
            ChatRequest(video_id="abc@def#ghi", query="test")
        assert "video_id" in str(exc.value)