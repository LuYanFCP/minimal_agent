from abc import ABC, abstractmethod
from typing import Any, TypedDict, TypeVar

MemoryType = TypeVar('MemoryType', bound='Memory')

class MemoryEntry(TypedDict):
    role: str  # user, assistant, system, tool, observation
    content: str
    timestamp: float
    metadata: dict[str, Any]


class Memory(ABC):
    @abstractmethod
    def add(self, entry: MemoryEntry) -> None:
        pass

    @abstractmethod
    def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class ListMemory(Memory):
    def __init__(self):
        self.entries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        return self.entries[-limit:] if self.entries else []

    def clear(self) -> None:
        self.entries = []
