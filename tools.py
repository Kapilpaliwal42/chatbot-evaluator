from pathlib import Path
import subprocess
import shlex

WORKSPACE = Path(r"D:\LocalAI\projects").resolve()



ALLOWED_COMMANDS = {
    "python",
    "git",
}


def run_command(command, timeout=10, max_output=10000):
    if not isinstance(command, str):
        raise ValueError("Command must be a string.")

    command = command.strip()

    if not command:
        raise ValueError("Command cannot be empty.")

    try:
        parts = shlex.split(command)
    except ValueError as e:
        raise ValueError(
            f"Invalid command syntax: {e}"
        )

    executable = parts[0].lower()

    if executable not in ALLOWED_COMMANDS:
        raise PermissionError(
            f"Command not allowed: {executable}"
        )

    try:
        result = subprocess.run(
            parts,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )

    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "exit_code": None,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds.",
            "timed_out": True,
        }

    stdout = result.stdout
    stderr = result.stderr

    if len(stdout) > max_output:
        stdout = (
            stdout[:max_output]
            + "\n...[stdout truncated]..."
        )

    if len(stderr) > max_output:
        stderr = (
            stderr[:max_output]
            + "\n...[stderr truncated]..."
        )

    return {
        "command": command,
        "exit_code": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": False,
    }

def resolve_workspace_path(path):
    path = Path(path)

    if not path.is_absolute():
        path = WORKSPACE / path

    path = path.resolve()

    try:
        path.relative_to(WORKSPACE)
    except ValueError:
        raise PermissionError(
            f"Access denied outside workspace: {path}"
        )

    return path


def list_files(directory="."):
    path = resolve_workspace_path(directory)

    if not path.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory}"
        )

    if not path.is_dir():
        raise NotADirectoryError(
            f"Not a directory: {directory}"
        )

    return [
        str(item.relative_to(path))
        for item in path.iterdir()
    ]


def read_file(file_path):
    path = resolve_workspace_path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    if not path.is_file():
        raise IsADirectoryError(
            f"Not a file: {file_path}"
        )

    return path.read_text(encoding="utf-8")

def write_file(file_path, content):
    path = resolve_workspace_path(file_path)

    if path.exists() and not path.is_file():
        raise IsADirectoryError(
            f"Not a file: {file_path}"
        )

    if path.exists():
        existing_content = path.read_text(
            encoding="utf-8"
        )

        if existing_content == content:
            return (
                f"No changes needed: {file_path} "
                "already contains the requested content."
            )

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        content,
        encoding="utf-8",
    )

    return (
        f"Successfully wrote {len(content)} characters "
        f"to {file_path}"
    )