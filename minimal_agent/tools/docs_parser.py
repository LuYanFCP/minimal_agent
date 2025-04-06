from collections.abc import Callable
import re
import inspect

from minimal_agent.tools.base import ToolDocsParser
from minimal_agent.tools.types import Arg, ToolDesc


class GoogleStyleDocsParser(ToolDocsParser):
    def parse(self, func: Callable, content: str) -> ToolDesc:
        func_name = func.__name__
        desc = ""
        args: list[Arg] = []

        pattern = re.compile(
            r"(?s)(?P<section>Args|Returns|Raises|Examples):\s*\n(?P<content>(.*?)(?:\n\s*))\n\s*(?!Returns|Raises|Yields|Examples|Note|Notes|Attributes|Todo|Warning|Warnings)",
            re.MULTILINE,
        )
        matches = pattern.finditer(content)

        description_match = re.match(
            r"^(?P<description>.*?)(?=\n\n|\Z)", content, re.DOTALL
        )
        if description_match:
            desc = description_match.group("description").strip()

        for match in matches:
            section = match.group("section")
            content_slice = match.group("content").strip()

            if section == "Args":
                args = self._parse_args(content_slice)

        args_name_map = {arg.arg_name: arg for arg in args}

        inspect_signature = inspect.signature(func)
        for param in inspect_signature.parameters.values():
            if param.name not in args_name_map:
                args.append(
                    Arg(
                        arg_name=param.name,
                        arg_desc="",
                        arg_type=(
                            param.annotation.__name__ if param.annotation else None
                        ),
                    )
                )
            else:
                args_name_map[param.name].arg_type = (
                    param.annotation.__name__ if param.annotation else None
                )

        return ToolDesc(
            name=func_name,
            description=desc,
            args=args,
        )

    def _parse_args(self, content: str) -> list[Arg]:
        args = []
        pattern = re.compile(
            r"(?P<key>.*?)\s*\s*(?:\((?P<type>.*?)\))?\s*:\s*(?P<desc>.+)"
        )

        for line in content.split("\n"):
            if line.strip():
                match_args = pattern.match(line)

                if match_args:
                    arg_name = match_args.group("key").strip()
                    arg_type = match_args.group("type")
                    arg_desc = match_args.group("desc").strip()
                    args.append(
                        Arg(
                            arg_name=arg_name,
                            arg_type=arg_type,
                            arg_desc=arg_desc,
                        )
                    )
        return args


