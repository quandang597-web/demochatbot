from dataclasses import dataclass, field
from llm_app.config import load_settings
from llm_app.llm_client import GeminiLLMClient, LLMClientProtocol, LLMError
from llm_app.logging_config import configure_logging

@dataclass
class ChatTurn:
    role: str
    text: str

@dataclass
class SessionState:
    history: list[ChatTurn] = field(default_factory=list)
    request_count: int = 0
    total_tokens: int = 0

    def clear(self) -> None:
        self.history.clear()
        self.request_count = 0
        self.total_tokens = 0

def handle_user_message(
    llm: LLMClientProtocol, state: SessionState, user_input: str
) -> str:
    cleaned = user_input.strip()
    if not cleaned:
        raise ValueError("Câu hỏi không được để trống.")

    result = llm.send_message(cleaned)
    state.history.append(ChatTurn(role="user", text=cleaned))
    state.history.append(ChatTurn(role="model", text=result.text))
    state.request_count += 1
    state.total_tokens += result.usage.total_tokens

    return (
        f"AI: {result.text}\n"
        f"[model={result.model}; latency={result.latency_ms:.0f} ms; "
        f"tokens={result.usage.total_tokens}]"
    )

def main() -> None:
    configure_logging()
    settings = load_settings()
    llm = GeminiLLMClient(settings)
    state = SessionState()

    print("=== LLM CHAT CLI ===")
    print("Lệnh: exit | clear | history | stats")

    try:
        while True:
            user_input = input("\nBạn: ").strip()
            if not user_input:
                print("Vui lòng nhập câu hỏi.")
                continue

            command = user_input.lower()
            if command in {"exit", "quit", "thoát"}:
                print("Đã kết thúc chương trình.")
                break

            if command == "clear":
                state.clear()
                llm.reset_chat()
                print("Đã xóa lịch sử hội thoại.")
                continue

            if command == "history":
                if not state.history:
                    print("Lịch sử trống.")
                else:
                    for turn in state.history[-20:]:
                        print(f"{turn.role.upper()}: {turn.text}")
                continue

            if command == "stats":
                print(
                    f"Thống kê phiên: Số request = {state.request_count} | "
                    f"Tổng token = {state.total_tokens}"
                )
                continue

            try:
                output = handle_user_message(llm, state, user_input)
                print(output)
            except (ValueError, LLMError) as error:
                print(f"Lỗi: {error}")
                continue
    finally:
        llm.close()

if __name__ == "__main__":
    main()