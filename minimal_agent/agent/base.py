import json
import re
from abc import ABC, abstractmethod

from pydantic import BaseModel

from minimal_agent.llm.base import LLMProviderType
from minimal_agent.memory.base import ListMemory, MemoryType
from minimal_agent.message import Message
from minimal_agent.tools.base import ToolType

# Add OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode


class AgentConfig(BaseModel):
    max_iterations: int = 10
    max_tokens: int = 8190
    temperature: float = 0.7
    timeout: float = 60.0
    model: str = "qwen-plus"


class AgentBase(ABC):
    def __init__(
        self,
        llm_provider: LLMProviderType,
        tools: list[ToolType] | None = None,
        memory: MemoryType | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self._llm = llm_provider
        self.tools = {tool._meta.name: tool for tool in tools or []}
        self.memory = memory or ListMemory()
        self.config = config or AgentConfig()
        self.state: dict = {"is_complete": False}
        # Initialize tracer for the base class
        self.tracer = trace.get_tracer("minimal_agent.agents.base")

    def reset(self) -> None:
        with self.tracer.start_as_current_span("agent.reset"):
            self.state = {"is_complete": False}

    def _get_tool_descriptions(self) -> str:
        with self.tracer.start_as_current_span("get_tool_descriptions") as span:
            span.set_attribute("tools.count", len(self.tools))

            if not self.tools:
                return "No tools available."

            descriptions = []
            for name, tool in self.tools.items():
                param = tool._meta.args
                params_desc = [
                    f"- {p.arg_name}: {p.arg_desc} ({p.arg_type}"
                    + (", required" if p.required else "")
                    + ")"
                    for p in param
                ]

                param_text = "\n".join(params_desc) if params_desc else "No parameters."

                descriptions.append(
                    f"Tool: {name}\n"
                    f"Description: {tool._meta.description}\n"
                    f"Parameters:\n{param_text}\n"
                )

            return "\n".join(descriptions)

    def _format_messages_from_memory(self, limit: int = 10) -> list[Message]:
        with self.tracer.start_as_current_span("format_messages_from_memory") as span:
            span.set_attribute("memory.limit", limit)

            entries = self.memory.get_recent(limit)
            messages = []

            for entry in entries:
                if entry["role"] in ["user", "assistant", "system"]:
                    messages.append(
                        Message(
                            role=entry["role"], 
                            content=entry["content"],
                            metadata=entry["metadata"]
                        )
                    )
                else:
                    messages.append(
                        Message(
                            role="assistant", 
                            content=entry["content"],
                            metadata=entry["metadata"]
                        )
                    )

            span.set_attribute("messages.count", len(messages))
            return messages

    def _parse_tool_call(self, content: str) -> dict | None:
        with self.tracer.start_as_current_span("parse_tool_call") as span:
            span.set_attribute("content.length", len(content))

            # Action: tool_name
            # Action Input: {"param1": "value1", "param2": "value2"}
            action_match = re.search(r"Action:\s*(\w+)", content)
            if not action_match:
                span.set_status(Status(StatusCode.ERROR, "No action match found"))
                return None

            tool_name = action_match.group(1)
            span.set_attribute("tool.name", tool_name)

            input_match = re.search(r"Action Input:\s*({.*?})", content, re.DOTALL)
            if not input_match:
                span.set_attribute("params.empty", True)
                return {"tool": tool_name, "params": {}}

            try:
                params = json.loads(input_match.group(1))
                span.set_attribute("params.count", len(params))
                return {"tool": tool_name, "params": params}
            except json.JSONDecodeError:
                span.set_status(
                    Status(StatusCode.ERROR, "JSON decode error for params")
                )
                params_text = input_match.group(1)
                params = {}

                for line in params_text.strip().split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        params[key.strip()] = value.strip()

                span.set_attribute("params.count", len(params))
                span.set_attribute("params.fallback_parsing", True)
                return {"tool": tool_name, "params": params}

    def _call_tool(self, tool_name: str, params: dict) -> str:
        with self.tracer.start_as_current_span("call_tool") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("params", str(params))

            if tool_name not in self.tools:
                error_msg = f"Tool '{tool_name}' not found. Available tools: {', '.join(self.tools.keys())}"
                span.set_status(Status(StatusCode.ERROR, error_msg))
                return f"Error: {error_msg}"

            tool = self.tools[tool_name]
            span.set_attribute("tool.description", tool._meta.description)

            try:
                result = tool(**params)
                span.set_attribute("result.length", len(str(result)))
                span.set_attribute("result", str(result))
                return f"Observation: {result}"
            except Exception as e:
                error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                span.set_status(Status(StatusCode.ERROR, error_msg))
                span.record_exception(e)
                return f"Error executing tool '{tool_name}': {str(e)}"

    async def _call_tool_async(self, tool_name: str, params: dict) -> str:
        with self.tracer.start_as_current_span("call_tool_async") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("params", str(params))

            if tool_name not in self.tools:
                error_msg = f"Tool '{tool_name}' not found. Available tools: {', '.join(self.tools.keys())}"
                span.set_status(Status(StatusCode.ERROR, error_msg))
                return f"Error: {error_msg}"

            tool = self.tools[tool_name]
            span.set_attribute("tool.description", tool._meta.description)

            try:
                result = await tool.execute_async(**params)
                span.set_attribute("result.length", len(str(result)))
                return f"Observation: {result}"
            except Exception as e:
                error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                span.set_status(Status(StatusCode.ERROR, error_msg))
                span.record_exception(e)
                return f"Error executing tool '{tool_name}': {str(e)}"

    @abstractmethod
    def run(self, input_text: str) -> str:
        pass

    @abstractmethod
    async def run_async(self, input_text: str) -> str:
        pass
