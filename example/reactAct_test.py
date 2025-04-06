import os
from uuid import uuid4
from minimal_agent.agent.react_agent import ReActAgent
from minimal_agent.llm.qwen import Qwen
from minimal_agent.memory.base import ListMemory
from minimal_agent.tools.websearch import SearxngWebSearch
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry import trace


resource = Resource(
    attributes={
        "service.name": "minimal_agent",
        "service.namespace": "react_agent",
        "service.instance.id": str(uuid4()),
    }
)

trace.set_tracer_provider(TracerProvider(resource=resource))

otlp_exporter = OTLPSpanExporter(endpoint=os.environ['OTLP_ENDPOINT'], insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)


agent = ReActAgent(
    llm_provider=Qwen(
        access_key=os.environ.get("QWEN_ACCESS_KEY"),
        model_name="qwen-plus",
    ),
    tools=[
        SearxngWebSearch(
            searx_host=os.environ.get('SEARXNG_HOST', 'http://localhost:8888'),
            count=3,
        )
    ],
    memory=ListMemory(),
)

print(agent.run("What is the trend of Alibaba's stock changes in the past 7 days?"))
