# llm_app/config.py
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str
    model_name: str
    timeout_seconds: float


def load_settings() -> Settings:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
    timeout_text = os.getenv("LLM_TIMEOUT_SECONDS", "60")

    if not api_key:
        raise RuntimeError(
            "Thiếu GEMINI_API_KEY. Hãy kiểm tra file .env."
        )

    timeout_seconds = float(timeout_text)
    if timeout_seconds <= 0:
        raise ValueError("LLM_TIMEOUT_SECONDS phải lớn hơn 0.")

    return Settings(
        api_key=api_key,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
    )