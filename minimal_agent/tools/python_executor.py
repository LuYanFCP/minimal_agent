import os
import re
from tempfile import template
from typing import NamedTuple, ParamSpec
import importlib
import traceback
from uuid import uuid4

from pydantic import BaseModel

from minimal_agent.tools.base import Tools, ToolsTypeEnum
from minimal_agent.tools.types import Arg

from RestrictedPython import compile_restricted
from RestrictedPython import safe_builtins
from RestrictedPython import limited_builtins
from RestrictedPython import utility_builtins
from RestrictedPython.PrintCollector import PrintCollector

class ImagePath(BaseModel):
    name: str
    url: str


class PltReplacedCode(NamedTuple):
    code: str
    images: list[ImagePath]


class PythonExecutorResult(BaseModel):
    output: str | None = None
    output_images: list[ImagePath] | None = None
    error: str | None = None


P = ParamSpec("P")

class PythonExecutor(Tools[P, str]):
    PLT_SAVE_PATTERN = re.compile(r"plt\.savefig\((.*)\)")
    def __init__(
            self,
            allow_module_set: frozenset[str] = frozenset(),
            is_allow_any: bool = False,
            storage_path: str = f"/tmp/{uuid4().hex}",
        ) -> None:
        super().__init__(
            name="python_executor",
            description="""
Execute Python code and return the result. This tool is designed to run Python code in a restricted environment. It allows you to import specific modules and execute code safely.
If you want to use matplotlib, please use plt.savefig(<image_name>) instead of plt.show(). The code will be modified to save the images to specified paths.
Action Input: ```python <code> ```
""",
            args=[
                Arg(
                    arg_name="code",
                    arg_desc="The Python code to execute.",
                    arg_type="str",
                    required=True,
                )

            ],
            func=self._inner_execute,
        )
        self._allow_module_set = allow_module_set
        self._is_allow_any = is_allow_any
        self._storage_path = storage_path

    def tool_type(self) -> ToolsTypeEnum:
        return ToolsTypeEnum.CODE_EXECUTOR

    def _inner_execute(self, code: str, ) -> PythonExecutorResult:
        """A function to execute Python code using RestrictedPython."""

        # Prepare a dictionary to store outputs
        output = []

        def _safe_import(name, *args, **kwargs):
            if not self._is_allow_any and name not in self._allow_module_set:
                raise ImportError(f"Importing module '{name}' is not allowed.")

            return __import__(name, *args, **kwargs)


        code_new, images = self._replace_plt_save(code)

        try:
            byte_code = compile_restricted(
                code_new,
                filename='<inline code>',
                mode='exec'
            )

            # Create a safe globals dictionary with allowed imports
            restricted_globals = {
                '__builtins__': {
                    **safe_builtins,
                    **limited_builtins,
                    **utility_builtins,
                    '_print_': PrintCollector,
                    "__import__": _safe_import,
                    "_getitem_": lambda x, y: x[y],
                    "_write_": lambda x: x
                },
            }
            local_vars = {}

            # Execute the code with restricted globals
            exec(byte_code, restricted_globals, local_vars)
            output = local_vars.get('_print', lambda: None)()

            return PythonExecutorResult(
                output=(output or "Code executed successfully with no output."),
                output_images=images or None,
            )

        except SyntaxError as e:
            return PythonExecutorResult(error=f"Syntax error: {str(e)}")
        except Exception as e:
            print(traceback.format_exc())
            return PythonExecutorResult(error=f"Error during execution: {str(e)}")

    def _replace_plt_save(self, code: str) -> PltReplacedCode:
        """Replace plt.show() to save images."""
        images = []
        code_lines = code.split('\n')
        new_code_lines = []

        for line in code_lines:
            match = self.PLT_SAVE_PATTERN.search(line)
            if match:
                image_name = match.group(1).strip().strip('"').strip("'")
                image_path = os.path.join(self._storage_path, image_name)
                images.append(ImagePath(name=image_name, url=f'file://{os.path.abspath(image_path)}'))
                line = line.replace(match.group(0), f"plt.savefig('{image_path}')")
            new_code_lines.append(line)

        return PltReplacedCode(code='\n'.join(new_code_lines), images=images)
