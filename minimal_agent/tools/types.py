from pydantic import BaseModel, Field


class Arg(BaseModel):
    arg_name: str
    arg_desc: str
    arg_type: str | None
    required: bool = Field(default=False, description="是否必填参数")


class ToolDesc(BaseModel):
    name: str = Field(description="tool name")
    description: str = Field(description="tool description")
    args: list[Arg] = Field(description="tools arguments")