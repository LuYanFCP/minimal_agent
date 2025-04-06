from datetime import datetime
from enum import StrEnum
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageTypeEnum(StrEnum):
    TEXT = "text"
    JSON = "json"


class Message(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
    )
    role: Literal["user", "system", "assistant"]
    content: str
    message_type: MessageTypeEnum = Field(
        default=MessageTypeEnum.TEXT,
    )
    metadata: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

