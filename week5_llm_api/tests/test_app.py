import os
import pytest
from llm_app.app import SessionState, handle_user_message
from llm_app.config import load_settings
from llm_app.llm_client import GeminiLLMClient, LLMError
from tests.fakes import FakeLLMClient

def test_handle_user_message_returns_fake_answer() -> None:
    client = FakeLLMClient(answer="LLM là mô hình ngôn ngữ lớn.")
    state = SessionState()
    output = handle_user_message(client, state, "LLM là gì?")
    assert "LLM là mô hình ngôn ngữ lớn." in output
    assert client.prompts == ["LLM là gì?"]
    assert state.request_count == 1
    assert state.total_tokens == 9
    assert len(state.history) == 2

def test_handle_user_message_rejects_empty_question() -> None:
    client = FakeLLMClient()
    state = SessionState()
    with pytest.raises(ValueError):
        handle_user_message(client, state, "   ")
    assert client.prompts == []
    assert state.request_count == 0

def test_failed_request_does_not_update_history() -> None:
    client = FakeLLMClient(should_fail=True)
    state = SessionState()
    with pytest.raises(LLMError):
        handle_user_message(client, state, "Câu hỏi gây lỗi")
    assert len(state.history) == 0
    assert state.total_tokens == 0

def test_session_clear_resets_state() -> None:
    client = FakeLLMClient()
    state = SessionState()
    handle_user_message(client, state, "Xin chào")
    state.clear()
    client.reset_chat()
    assert len(state.history) == 0
    assert state.request_count == 0
    assert state.total_tokens == 0
    assert client.reset_called == 1

@pytest.mark.skipif(
    os.getenv("RUN_LLM_INTEGRATION_TESTS") != "1",
    reason="Không bật test gọi API thật",
)
def test_real_llm_returns_text() -> None:
    settings = load_settings()
    client = GeminiLLMClient(settings)
    try:
        result = client.generate("Trả lời đúng một từ: Xin chào")
        assert result.text.strip()
        assert result.latency_ms > 0
    finally:
        client.close()