import pytest
from llm_app.config import load_settings

def test_load_settings_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-123")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    settings = load_settings()
    assert settings.api_key == "fake-key-123"
    assert settings.model_name == "gemini-2.5-flash"
    assert settings.timeout_seconds == 30.0

def test_load_settings_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(RuntimeError):
        load_settings()

def test_load_settings_invalid_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-123")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "-5")
    with pytest.raises(ValueError):
        load_settings()