from dataclasses import dataclass, field

@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thoughts_tokens: int = 0

@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    latency_ms: float
    usage: TokenUsage = field(default_factory=TokenUsage)
    request_id: str = ""