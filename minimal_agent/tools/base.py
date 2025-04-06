from abc import ABC
from collections.abc import Callable
from typing import Generic, TypeVar, ParamSpec
from minimal_agent.tools.types import ToolDesc, Arg

ValueType = TypeVar("ValueType")
ToolType = TypeVar('ToolType', bound='Tools')

class ToolDocsParser(ABC):
    def parse(self, func: Callable, content: str) -> ToolDesc: ...

P = ParamSpec("P")

class Tools(Generic[P, ValueType]):
    def __init__(
        self, name: str, description: str, args: list[Arg], func: Callable[P, ValueType]
    ) -> None:
        self._meta = ToolDesc(
            name=name,
            description=description,
            args=args,
        )
        self._func = func

    @classmethod
    def create_tool(
        cls,
        func: Callable[P, ValueType],
        parser: ToolDocsParser,
    ) -> "Tools":
        doc = func.__doc__ or ""
        tool_desc = parser.parse(func, doc)
        return cls(
            name=tool_desc.name,
            description=tool_desc.description,
            args=tool_desc.args,
            func=func,
        )

    def __repr__(self) -> str:
        return f"Tools(name={self._meta.name}, description={self._meta.description}, args={self._meta.args})"

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> ValueType:
        return self._func(*args, **kwargs)
