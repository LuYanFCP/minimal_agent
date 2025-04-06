import json
from typing import Any, Literal

import dashscope
from minimal_agent.llm.base import LLMProvider
from dashscope.api_entities.dashscope_response import (
    Message as DashScopeMsg,
)
from minimal_agent.message import Message, MessageTypeEnum
from opentelemetry import trace

QwenModelLiteral = Literal["qwen-max", "qwen-plus"]


class Qwen(LLMProvider):
    def __init__(
        self,
        access_key: str,
        model_name: str,
    ) -> None:
        super().__init__(model_name)
        self.__access_key = access_key
        self.tracer = trace.get_tracer("minimal_agent.llm.qwen")

    def completion(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float = 0.9,
        stop: list[str] | str | None = None,
        response_format: Literal["text", "json"] = "text",
        **kwargs: dict[str, Any],
    ) -> Message:
        input = [
            DashScopeMsg(role=item.role, content=item.content) for item in messages
        ]
        with self.tracer.start_as_current_span("qwen.completion") as span:
            span.set_attribute("model_name", self.model_name)
            span.set_attribute("messages.count", len(messages))
            span.set_attribute("temperature", temperature)
            span.set_attribute("max_tokens", max_tokens)
            span.set_attribute("top_p", top_p)
            span.set_attribute("stop", stop)
            span.set_attribute("response_format", response_format)
            span.set_attribute("input", json.dumps(input))
            response = dashscope.Generation.call(
                api_key=self.__access_key,
                model=self.model_name,
                messages=input,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                response_format={
                    "type": "json_object" if response_format == "json" else response_format
                },
                **kwargs,
            )

            span.set_attribute("response", response.output.text)
            return Message(
                role="assistant",
                content=response.output.text,
                message_type=MessageTypeEnum(response_format),
                metadata={
                    'model': self.model_name,
                }
            )

    async def completion_async(
        self, 
        messages: list[Message], 
        temperature = 0.7, 
        max_tokens = None, 
        top_p = 0.9, 
        stop = None, 
        response_format = "text", 
        **kwargs
    ) -> Message:
        raise NotImplementedError("Qwen LLM does not support async completion.")