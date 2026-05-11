from pathlib import Path
import functools
import importlib
import inspect

def load_tools(logging=print, names=[], catalog=None):
    """Load every tool module under ``agent.tools``.

    F-class: when ``catalog`` is provided, tool modules whose ``tool_info``
    or ``tool_function`` declares a ``catalog`` parameter receive the catalog
    bound via ``functools.partial`` (function) or by direct keyword call
    (info). Tools without a ``catalog`` parameter (bash, edit) are unaffected.
    """
    tools_dir = Path(__file__).parent
    tools = []

    # Get all Python files in the tools directory (excluding __init__.py)
    tool_files = [f for f in tools_dir.glob("*.py") if f.stem != "__init__"]

    for tool_file in tool_files:
        # Import the module
        module_name = f"agent.tools.{tool_file.stem}"
        try:
            module = importlib.import_module(module_name)

            # Check if module has required functions
            if hasattr(module, 'tool_info') and hasattr(module, 'tool_function'):
                tool_name = tool_file.stem
                if names and (names == 'all' or tool_name in names):
                    fn = module.tool_function
                    info_fn = module.tool_info
                    if 'catalog' in inspect.signature(fn).parameters:
                        fn = functools.partial(fn, catalog=catalog)
                    if 'catalog' in inspect.signature(info_fn).parameters:
                        info = info_fn(catalog=catalog)
                    else:
                        info = info_fn()
                    tools.append({'info': info, 'function': fn, 'name': tool_name})
            else:
                raise Exception(f"Tool module {module_name} does not have required functions.")
        except Exception as e:
            # Log the error and raise it
            logging(f"Failed to import {module_name}: {e}")
            raise e

    return tools
