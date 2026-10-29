from __future__ import annotations
import logging
from time import perf_counter
from typing import Protocol
from uuid import uuid4
from google import genai
from google.genai import types
from llm_app.config import Settings
from llm_app.schemas import LLMResponse, TokenUsage

logger = logging.getLogger(__name__)

class LLMError(RuntimeError):
    """Lỗi đã được chuẩn hóa ở tầng ứng dụng."""

class LLMClientProtocol(Protocol):
    def generate(self, prompt: str) -> LLMResponse: ...
    def send_message(self, prompt: str) -> LLMResponse: ...
    def reset_chat(self) -> None: ...

def parse_usage(metadata: object | None) -> TokenUsage:
    if metadata is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=int(getattr(metadata, "prompt_token_count", 0) or 0),
        output_tokens=int(getattr(metadata, "candidates_token_count", 0) or 0),
        total_tokens=int(getattr(metadata, "total_token_count", 0) or 0),
        thoughts_tokens=int(getattr(metadata, "thoughts_token_count", 0) or 0),
    )

class GeminiLLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = genai.Client(api_key=settings.api_key)
        self._config = types.GenerateContentConfig(
            system_instruction=(
                "Bạn là trợ giảng LLM. Trả lời tiếng Việt, "
                "rõ ràng và không tự tạo số liệu."
            ),
            temperature=0.3,
            max_output_tokens=800,
        )
        self._chat = self._client.chats.create(
            model=self._settings.model_name,
            config=self._config,
        )

    def close(self) -> None:
        self._client.close()

    def reset_chat(self) -> None:
        """Tạo phiên chat mới để xóa sạch ngữ cảnh phía SDK."""
        self._chat = self._client.chats.create(
            model=self._settings.model_name,
            config=self._config,
        )

    def generate(self, prompt: str) -> LLMResponse:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Prompt không được để trống.")

        request_id = str(uuid4())
        started = perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self._settings.model_name,
                contents=cleaned_prompt,
                config=self._config,
            )
        except Exception as error:
            logger.error("llm_request error request_id=%s", request_id)
            raise LLMError(f"Không thể gọi LLM API. request_id={request_id}") from error

        return self._build_response(response, started, request_id)

    def send_message(self, prompt: str) -> LLMResponse:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Prompt không được để trống.")

        request_id = str(uuid4())
        started = perf_counter()
        try:
            response = self._chat.send_message(cleaned_prompt)
        except Exception as error:
            logger.error("llm_chat error request_id=%s", request_id)
            raise LLMError(f"Không thể gọi LLM API. request_id={request_id}") from error

        return self._build_response(response, started, request_id)

    def _build_response(self, response: object, started: float, request_id: str) -> LLMResponse:
        latency_ms = (perf_counter() - started) * 1000
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise LLMError(f"Mô hình không trả text. request_id={request_id}")

        usage = parse_usage(getattr(response, "usage_metadata", None))
        result = LLMResponse(
            text=text,
            model=self._settings.model_name,
            latency_ms=latency_ms,
            usage=usage,
            request_id=request_id,
        )
        logger.info(
            "llm_request success request_id=%s model=%s latency_ms=%.1f total_tokens=%d",
            result.request_id,
            result.model,
            result.latency_ms,
            result.usage.total_tokens,
        )
        return result