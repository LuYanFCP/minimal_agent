# Minimal Agent

A lightweight, educational framework for learning AI agent development with tools integration. (Use claude3.7 when writing code)

## Purpose

This project serves as a learning tool for those interested in:
- Understanding the core concepts of AI agents
- Learning how to integrate Large Language Models with tools
- Exploring agent architectures and reasoning patterns
- Gaining hands-on experience with agent implementation

The minimalist design intentionally focuses on readability and conceptual clarity rather than advanced features or production readiness.

## Requirements

- **Python 3.13**: This framework specifically targets Python 3.13 and uses its latest features
- A supported LLM provider (Qwen, OpenAI, etc.)

## Features

- Simple, modular architecture that's easy to understand
- Support for various LLM providers (Qwen, OpenAI, etc.)
- Basic tool integration system with ReAct pattern
- Memory systems for context management
- OpenTelemetry integration for observability and learning monitoring concepts

## Installation

Ensure you have Python 3.13 installed before proceeding:

```bash
python --version  # Should show Python 3.13.x
```

Then install the package:

```bash
# Install with PDM
pdm install

# Development setup
pdm install --dev
```

## Usage Example

```python
from minimal_agent.agent.react_agent import ReActAgent
from minimal_agent.llm.qwen import Qwen
from minimal_agent.memory.base import ListMemory
from minimal_agent.tools.websearch import SearxngWebSearch

# Initialize the agent with tools
agent = ReActAgent(
    llm_provider=Qwen(
        access_key="your-api-key",
        model_name="qwen-plus",
    ),
    tools=[
        SearxngWebSearch(
            searx_host="http://localhost:8888",
            count=3,
        )
    ],
    memory=ListMemory(),
)

# Run the agent
result = agent.run("What is the trend of Alibaba's stock changes in the past 7 days?")
print(result)

# output:
"""
Based on the historical price data from Yahoo Finance, here are Alibaba's (BABA) stock closing prices for the past 7 days:

- Apr 4, 2025: $116.54
- Apr 3, 2025: $129.33
- Apr 2, 2025: $129.79
- Apr 1, 2025: $132.70
- Mar 31, 2025: $132.23
- Mar 28, 2025: $132.43
- Mar 27, 2025: $135.63

During this period, we observe that Alibaba's stock price has decreased from $135.63 on Mar 27, 2025, to $116.54 on Apr 4, 2025, representing a decline of approximately 14.08%. The overall trend indicates a downward movement in Alibaba's stock price over the past 7 days.

Reference:
[Alibaba Group Holding Limited (BABA) Stock Historical Prices & Data](https://finance.yahoo.com/quote/BABA/history) by Unknown, published on 2025-04-06 and accessed on 2025-04-06.
"""

```

## Learning Architecture

The framework follows a modular design with these core components to help you understand agent systems:

- **Agent**: Learn how reasoning loops work and how tools are integrated into decision making
- **LLM Provider**: Understand how to abstract different language model services 
- **Tools**: See how reusable capabilities are implemented and connected to agents
- **Memory**: Explore how conversation context can be maintained across interactions

## TODO:

- [ ]: implement PlanAgent.
- [ ]: implement HybridAgent.
- [ ]: Support MCP.
- [ ]: implemet a simple deep search.
- [ ]: add web ui for agent.

## License

MIT
