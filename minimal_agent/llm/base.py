from typing import Any, Literal, TypeVar
from pydantic import BaseModel
from abc import ABC, abstractmethod
from minimal_agent.message import Message

LLMProviderType = TypeVar('LLMProviderType', bound='LLMProvider')

class CompletionsOptions(BaseModel):
    model: str | None = None


class LLMProvider(ABC):

    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name

    @abstractmethod
    def completion(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float = 0.9,
        stop: list[str] | str | None = None,
        response_format: Literal['text', 'json'] = "text",
        **kwargs: dict[str, Any],
    ) -> Message:
        pass

    @abstractmethod
    async def completion_async(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        response_format: Literal['text', 'json'] = "text",
        **kwargs: dict[str, Any],
    ) -> Message:
        pass

