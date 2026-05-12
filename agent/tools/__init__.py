from pathlib import Path
import functools
import importlib
import inspect

def _bind_current_gen(fn, current_gen):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        kwargs.pop("current_gen", None)
        return fn(*args, current_gen=current_gen, **kwargs)

    return wrapper


def load_tools(logging=print, names=[], catalog=None, workspace_root=None, current_gen=None):
    """Load every tool module under ``agent.tools``.

    When ``catalog`` is provided, tool modules whose ``tool_info`` or
    ``tool_function`` declares a ``catalog`` parameter receive the catalog
    bound via ``functools.partial`` (function) or by direct keyword call
    (info). Tools without a ``catalog`` parameter (bash, edit) are unaffected.

    ``workspace_root`` and ``current_gen`` are bound the same way for any
    tool that declares them. Neither is exposed in tool schemas; if a
    model includes one anyway, the wrapper discards that value and uses
    the harness-provided value.

    Modules whose filename begins with ``_`` are treated as private
    helpers (e.g. shared safety code) and skipped; this lets the tool
    package keep helper modules colocated without each one having to
    export a sham ``tool_info`` / ``tool_function`` pair.
    """
    tools_dir = Path(__file__).parent
    tools = []

    # Get all Python files in the tools directory (excluding __init__.py
    # and private underscore-prefixed helper modules).
    tool_files = [
        f for f in tools_dir.glob("*.py")
        if f.stem != "__init__" and not f.stem.startswith("_")
    ]

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
                    fn_params = inspect.signature(fn).parameters
                    info_params = inspect.signature(info_fn).parameters
                    fn_binds = {}
                    info_binds = {}
                    if 'catalog' in fn_params:
                        fn_binds['catalog'] = catalog
                    if 'catalog' in info_params:
                        info_binds['catalog'] = catalog
                    if 'workspace_root' in fn_params:
                        fn_binds['workspace_root'] = workspace_root
                    if 'workspace_root' in info_params:
                        info_binds['workspace_root'] = workspace_root
                    if fn_binds:
                        fn = functools.partial(fn, **fn_binds)
                    if 'current_gen' in fn_params:
                        fn = _bind_current_gen(fn, current_gen)
                    info = info_fn(**info_binds) if info_binds else info_fn()
                    tools.append({'info': info, 'function': fn, 'name': tool_name})
            else:
                raise Exception(f"Tool module {module_name} does not have required functions.")
        except Exception as e:
            # Log the error and raise it
            logging(f"Failed to import {module_name}: {e}")
            raise e

    return tools
