from llm_app.llm_client import LLMError
from llm_app.schemas import LLMResponse, TokenUsage

class FakeLLMClient:
    def __init__(self, answer: str = "Câu trả lời mẫu", should_fail: bool = False) -> None:
        self.answer = answer
        self.should_fail = should_fail
        self.prompts: list[str] = []
        self.reset_called: int = 0

    def generate(self, prompt: str) -> LLMResponse:
        return self.send_message(prompt)

    def send_message(self, prompt: str) -> LLMResponse:
        if self.should_fail:
            raise LLMError("Lỗi giả lập từ FakeLLMClient")
        self.prompts.append(prompt)
        return LLMResponse(
            text=self.answer,
            model="fake-model",
            latency_ms=1.0,
            usage=TokenUsage(prompt_tokens=5, output_tokens=4, total_tokens=9),
            request_id="test-request-id",
        )

    def reset_chat(self) -> None:
        self.reset_called += 1