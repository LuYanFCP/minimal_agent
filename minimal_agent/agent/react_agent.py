import asyncio
from datetime import datetime
import re
from .base import AgentBase
from minimal_agent.llm.base import LLMProviderType
from minimal_agent.memory.base import MemoryType
from minimal_agent.tools.base import ToolType
from minimal_agent.message import Message

# Add OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode


class ReActAgent(AgentBase):
    def __init__(
        self,
        llm_provider: LLMProviderType,
        tools: list[ToolType] | None = None,
        memory: MemoryType | None = None,
    ) -> None:
        super().__init__(llm_provider, tools, memory)
        self.mode = "REACT"
        # Get tracer for this class
        self.tracer = trace.get_tracer("minimal_agent.agents.react")

    def _create_react_prompt(self) -> str:
        return (
f"""
# Task

You are a helpful AI assistant that can use tools to solve problems step by step.


# Tools

You can use the following tools:


{self._get_tool_descriptions()}


# Instructions

1. Think about the problem step by step
2. When you need to use a tool, use the following format:
   Thought: <your reasoning about what to do>
   Action: <tool_name>
   Action Input: <tool parameters in JSON format>
3. Tools will respond with:
   Observation: <tool result>
4. After receiving an observation, continue your reasoning
5. When you have a final answer, respond with:
   Thought: <your final reasoning>
   Answer: <your final answer>


# Important Rules

- ALWAYS follow the Thought/Action/Observation/Answer format
- If using a tool, only one tool at a time.
- NEVER make up tool results
- If a tool fails, try a different approach
- Be thorough and detailed in your reasoning
- If you can't find an answer, say "I don't know" instead of making something up
- If python execution is exist, you can generate code and execute it.
- Do NOT provide an Answer if you are uncertain or unable to complete all required actions
- If you cannot fully solve the problem, use your memory to explain your limitations and what additional information or tools you would need to complete it.
- Additionally, please include a reference to the original article at the end of your summary. The reference should be formatted as follows:

[Article Title](URL) by [Author Name], published on [Publication Date] and accessed on [Access Date].
Make sure to use proper Markdown syntax for headings, lists, and the hyperlink in the reference. Here is an example of how the reference should look:

> [The Impact of AI on Society](https://www.example.com/ai-impact-society) by John Doe, published on 2023-06-15 and accessed on 2025-04-06.

"""

        )

    def run(self, input_text: str) -> str:
        # Create a span for the entire run operation
        with self.tracer.start_as_current_span("react_agent.run") as run_span:
            run_span.set_attribute("input.text", input_text)

            self.reset()
            self.state["input"] = input_text

            with self.tracer.start_as_current_span("create_system_prompt"):
                system_prompt = self._create_react_prompt()

            self.memory.add(
                {
                    "role": "system",
                    "content": system_prompt,
                    "timestamp": datetime.now().timestamp(),
                    "metadata": {},
                }
            )

            self.memory.add(
                {
                    "role": "user",
                    "content": input_text,
                    "timestamp": datetime.now().timestamp(),
                    "metadata": {},
                }
            )

            iterations = 0
            while iterations < self.config.max_iterations and not self.state.get(
                "is_complete", False
            ):
                iterations += 1
                self.state["current_step"] = iterations

                # Create a span for each iteration
                with self.tracer.start_as_current_span(
                    f"iteration_{iterations}"
                ) as iteration_span:
                    iteration_span.set_attribute("iteration.number", iterations)

                    with self.tracer.start_as_current_span("format_messages"):
                        messages = self._format_messages_from_memory()

                    with self.tracer.start_as_current_span(
                        "llm_completion"
                    ) as llm_span:
                        llm_span.set_attribute("temperature", self.config.temperature)
                        llm_span.set_attribute("max_tokens", self.config.max_tokens)

                        response = self._llm.completion(
                            messages=messages,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                        )

                        assistant_message = response.content
                        llm_span.set_attribute(
                            "response_length", len(assistant_message)
                        )

                    self.memory.add(
                        {
                            "role": "assistant",
                            "content": assistant_message,
                            "timestamp": datetime.now().timestamp(),
                            "metadata": {"step": iterations},
                        }
                    )

                    if "Answer:" in assistant_message and 'Action:' not in assistant_message:
                        with self.tracer.start_as_current_span("extract_answer"):
                            answer_match = re.search(
                                r"Answer:\s*(.*?)(?:$|Thought:)",
                                assistant_message,
                                re.DOTALL,
                            )
                            if answer_match:
                                self.state["response"] = answer_match.group(1).strip()
                                self.state["is_complete"] = True
                                iteration_span.set_attribute("found_answer", True)
                                break

                    with self.tracer.start_as_current_span("parse_tool_call"):
                        tool_call = self._parse_tool_call(assistant_message)

                    if tool_call:
                        with self.tracer.start_as_current_span(
                            "tool_execution"
                        ) as tool_span:
                            tool_span.set_attribute("tool.name", tool_call["tool"])
                            tool_span.set_attribute(
                                "tool.params", str(tool_call["params"])
                            )

                            observation = self._call_tool(
                                tool_call["tool"], tool_call["params"]
                            )
                            tool_span.set_attribute(
                                "observation_length", len(observation)
                            )

                            self.memory.add(
                                {
                                    "role": "observation",
                                    "content": observation,
                                    "timestamp": datetime.now().timestamp(),
                                    "metadata": {
                                        "step": iterations,
                                        "tool": tool_call["tool"],
                                    },
                                }
                            )
                    else:
                        if iterations >= self.config.max_iterations:
                            iteration_span.set_status(
                                Status(StatusCode.ERROR, "Max iterations reached")
                            )
                            self.state["response"] = (
                                "I'm sorry, I couldn't complete the task within the allowed iterations."
                            )
                            self.state["is_complete"] = True
                            break

            # Set span status based on completion
            if not self.state.get("is_complete", False):
                run_span.set_status(Status(StatusCode.ERROR, "Failed to complete task"))
                self.state["response"] = (
                    "I'm sorry, I couldn't complete the task within the allowed iterations."
                )
                self.state["is_complete"] = True
            else:
                run_span.set_status(Status(StatusCode.OK))
                run_span.set_attribute("iterations.total", iterations)

        return self.state["response"]

    async def run_async(self, input_text: str) -> str:
        """运行 ReAct Agent (异步方法)"""
        # Create a span for the entire async run operation
        with self.tracer.start_as_current_span("react_agent.run_async") as run_span:
            run_span.set_attribute("input.text", input_text)

            self.reset()
            self.state["input"] = input_text

            with self.tracer.start_as_current_span("create_system_prompt"):
                system_prompt = self._create_react_prompt()

            self.memory.add(
                {
                    "role": "system",
                    "content": system_prompt,
                    "timestamp": datetime.now().timestamp(),
                    "metadata": {},
                }
            )

            self.memory.add(
                {
                    "role": "user",
                    "content": input_text,
                    "timestamp": datetime.now().timestamp(),
                    "metadata": {},
                }
            )

            iterations = 0
            while iterations < self.config.max_iterations and not self.state.get(
                "is_complete", False
            ):
                iterations += 1
                self.state["current_step"] = iterations

                # Create a span for each iteration
                with self.tracer.start_as_current_span(
                    f"iteration_{iterations}"
                ) as iteration_span:
                    iteration_span.set_attribute("iteration.number", iterations)

                    with self.tracer.start_as_current_span("format_messages"):
                        messages = self._format_messages_from_memory()

                    with self.tracer.start_as_current_span(
                        "llm_completion_async"
                    ) as llm_span:
                        llm_span.set_attribute("temperature", self.config.temperature)
                        llm_span.set_attribute("max_tokens", self.config.max_tokens)

                        if hasattr(self.llm_client, "create_chat_completion_async"):
                            response = (
                                await self.llm_client.create_chat_completion_async(
                                    messages=messages,
                                    temperature=self.config.temperature,
                                    max_tokens=self.config.max_tokens,
                                )
                            )
                            assistant_message = response.content
                        else:
                            response = await asyncio.to_thread(
                                self._llm.completion,
                                messages=messages,
                                temperature=self.config.temperature,
                                max_tokens=self.config.max_tokens,
                            )
                            assistant_message = response.content

                        llm_span.set_attribute(
                            "response_length", len(assistant_message)
                        )

                    self.memory.add(
                        {
                            "role": "assistant",
                            "content": assistant_message,
                            "timestamp": datetime.now().timestamp(),
                            "metadata": {"step": iterations},
                        }
                    )

                    if "Answer:" in assistant_message:
                        with self.tracer.start_as_current_span("extract_answer"):
                            answer_match = re.search(
                                r"Answer:\s*(.*?)(?:$|Thought:)",
                                assistant_message,
                                re.DOTALL,
                            )
                            if answer_match:
                                self.state["response"] = answer_match.group(1).strip()
                                self.state["is_complete"] = True
                                iteration_span.set_attribute("found_answer", True)
                                break

                    with self.tracer.start_as_current_span("parse_tool_call"):
                        tool_call = self._parse_tool_call(assistant_message)

                    if tool_call:
                        with self.tracer.start_as_current_span(
                            "tool_execution_async"
                        ) as tool_span:
                            tool_span.set_attribute("tool.name", tool_call["tool"])
                            tool_span.set_attribute(
                                "tool.params", str(tool_call["params"])
                            )

                            observation = await self._call_tool_async(
                                tool_call["tool"], tool_call["params"]
                            )
                            tool_span.set_attribute(
                                "observation_length", len(observation)
                            )

                            self.memory.add(
                                {
                                    "role": "observation",
                                    "content": observation,
                                    "timestamp": datetime.now().timestamp(),
                                    "metadata": {
                                        "step": iterations,
                                        "tool": tool_call["tool"],
                                    },
                                }
                            )
                    else:
                        if iterations >= self.config.max_iterations:
                            iteration_span.set_status(
                                Status(StatusCode.ERROR, "Max iterations reached")
                            )
                            self.state["response"] = (
                                "I'm sorry, I couldn't complete the task within the allowed iterations."
                            )
                            self.state["is_complete"] = True
                            break

            # Set span status based on completion
            if not self.state.get("is_complete", False):
                run_span.set_status(Status(StatusCode.ERROR, "Failed to complete task"))
                self.state["response"] = (
                    "I'm sorry, I couldn't complete the task within the allowed iterations."
                )
                self.state["is_complete"] = True
            else:
                run_span.set_status(Status(StatusCode.OK))
                run_span.set_attribute("iterations.total", iterations)

        return self.state["response"]
