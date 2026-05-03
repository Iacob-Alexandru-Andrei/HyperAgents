import asyncio
import os
import signal
import time


_MAX_OUTPUT_BYTES = 20_000


def _truncate_output(text, label):
    data = text.encode("utf-8", errors="replace")
    if len(data) <= _MAX_OUTPUT_BYTES:
        return text
    truncated = data[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    omitted = len(data) - _MAX_OUTPUT_BYTES
    return f"{truncated}\n... [{label} truncated: {omitted} more bytes omitted]"

def tool_info():
    return {
        "name": "bash",
        "description": """Run commands in a bash shell
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
* You don't have access to the internet via this tool.
* You do have access to a mirror of common linux and python packages via apt and pip.
* State is persistent across command calls and discussions with the user.
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
* Please avoid commands that may produce a very large amount of output.
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run."
                },
                "interactive": {
                    "type": "boolean",
                    "description": "Start bash in interactive mode. Defaults to false."
                }
            },
            "required": ["command"]
        }
    }

class BashSession:
    """A session of a bash shell."""
    def __init__(self, interactive=False):
        self._started = False
        self._process = None
        self._pgid = None
        self._interactive = interactive
        self._timed_out = False
        self._timeout = 120.0  # seconds
        self._sentinel = "<<exit>>"
        self._output_delay = 0.2  # seconds

    async def start(self):
        if self._started:
            return
        self._process = await asyncio.create_subprocess_shell(
            "/bin/bash -i" if self._interactive else "/bin/bash",
            start_new_session=True,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy()  # Ensures inheritance of the current environment
        )
        self._pgid = self._process.pid
        self._started = True

    def _signal_process_group(self, sig):
        if self._pgid is None:
            return
        try:
            os.killpg(self._pgid, sig)
        except ProcessLookupError:
            return

    async def _kill_process_group(self):
        if self._process is None:
            return
        self._signal_process_group(signal.SIGTERM)
        try:
            await asyncio.wait_for(self._process.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            self._signal_process_group(signal.SIGKILL)
            try:
                await asyncio.wait_for(self._process.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass
        self._process = None
        self._pgid = None
        self._started = False

    def stop(self):
        if not self._started:
            return
        if self._process.returncode is None:
            self._signal_process_group(signal.SIGTERM)
            time.sleep(0.5)
            if self._process.returncode is None:
                self._signal_process_group(signal.SIGKILL)
        self._process = None
        self._pgid = None
        self._started = False

    async def run(self, command):
        if not self._started:
            raise ValueError("Session has not started.")
        if self._process.returncode is not None:
            raise ValueError(f"Bash has exited with returncode {self._process.returncode}")
        if self._timed_out:
            raise ValueError(
                f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
            )
        
        # Send command
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # Read output until sentinel
        try:
            output = ''
            start_time = asyncio.get_event_loop().time()
            
            while True:
                if asyncio.get_event_loop().time() - start_time > self._timeout:
                    self._timed_out = True
                    await self._kill_process_group()
                    raise ValueError(
                        f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
                    )
                
                await asyncio.sleep(self._output_delay)
                # Read from the internal buffer
                stdout_data = self._process.stdout._buffer.decode(errors='ignore')
                stderr_data = self._process.stderr._buffer.decode(errors='ignore')
                
                if self._sentinel in stdout_data:
                    output = stdout_data[: stdout_data.index(self._sentinel)]
                    break

            # Clear buffers
            self._process.stdout._buffer.clear()
            self._process.stderr._buffer.clear()

            output = _truncate_output(output.strip(), "stdout")
            error = _truncate_output(stderr_data.strip(), "stderr")

            return output, error

        except Exception as e:
            self._timed_out = True
            raise ValueError(str(e))

def filter_error(error):
    # Filter out errors that we do not want to see
    filtered_lines = []
    i = 0
    error_lines = error.splitlines()
    while i < len(error_lines):
        line = error_lines[i]

        # Skip the next lines if ioctl error, add relevant lines
        if "Inappropriate ioctl for device" in line:
            i += 3
            if '<<exit>>' in error_lines[i]:
                i += 1
            while i < len(error_lines) - 1:
                filtered_lines.append(error_lines[i])
                i += 1
            i += 1
            continue

        filtered_lines.append(line)
        i += 1
    return '\n'.join(filtered_lines).strip()

async def tool_function_call(command, interactive=False):
    """Execute a command in the bash shell."""
    try:
        bash_session = BashSession(interactive=interactive)

        if not bash_session._started:
            await bash_session.start()

        output, error = await bash_session.run(command)
        error = filter_error(error)
        result = ""
        if output:
            result += output
        if error:
            result += "\nError:\n" + error
        return result.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def tool_function(command, interactive=False):
    return asyncio.run(tool_function_call(command, interactive=interactive))

if __name__ == "__main__":
    # Example usage
    import sys

    # Check if the script is called with arguments
    if len(sys.argv) < 2:
        print("Usage: python bash.py '<command>'")
    else:
        # Extract the command from the command-line arguments
        input_command = ' '.join(sys.argv[1:])
        # Run the tool_function asynchronously
        result = tool_function(input_command)
        print(result)
