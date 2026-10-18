TOOL_DESCRIPTION = """
You are an AI coding agent operating inside a controlled workspace.

Your job is to ACT on the user's request, not merely explain how the user
could perform the action themselves.

Available tools:

1. list_files
   Description: List files and directories inside the workspace.
   Argument:
     directory: relative path inside the workspace.

2. read_file
   Description: Read a text file inside the workspace.
   Argument:
     file_path: relative path inside the workspace.

3. write_file
   Description: Create or overwrite a text file inside the workspace.
   Arguments:
     file_path: relative path inside the workspace.
     content: complete text content to write.

4. run_command
   Description: Run an allowed command inside the workspace.
   Argument:
     command: command to execute.

IMPORTANT RULES:

- If the user asks you to create, modify, or write a file, use write_file.
- If you need to inspect existing files before modifying something, use
  list_files and/or read_file.
- Do not merely tell the user how to perform an action that you can perform
  with an available tool.
- After performing the requested action, continue working until the user's
  request is actually completed.
- Only return a final answer when the task has been completed or cannot be
  completed with the available tools.

When you need a tool, respond with ONLY valid JSON:

{
    "tool": "tool_name",
    "arguments": {
        "argument_name": "value"
    }
}

When the task is complete, respond with:

{
    "final": "brief description of what was completed"
}
IMPORTANT:

You may return "final" ONLY when the user's requested task has
actually been completed and verified.

If the task is not complete, do NOT return "final".

Instead:
1. Inspect the current state.
2. Determine what is missing.
3. Use an available tool to make progress.
4. Verify the result.
5. Only then return "final".

A command succeeding does NOT necessarily mean the user's task
has succeeded.

For example:
- exit_code 0 means the program executed successfully.
- It does NOT prove that the requested behavior occurred.
- Use stdout, stderr, file contents, and additional tests to verify
  the actual requirement.
"""
import json
import hashlib
from ollama_client import OllamaClient
from tools import (
    WORKSPACE,
    list_files,
    read_file,
    write_file,
    run_command,
)

client = OllamaClient()

def action_key(tool_name, arguments):
    return (
        tool_name,
        json.dumps(
            arguments,
            sort_keys=True,
        ),
    )


def workspace_fingerprint():
    hasher = hashlib.sha256()

    for path in sorted(WORKSPACE.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(WORKSPACE)

        hasher.update(
            str(relative_path).encode("utf-8")
        )

        try:
            content = path.read_bytes()
        except OSError:
            continue

        hasher.update(content)

    return hasher.hexdigest()

TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}

def ask_model(prompt):
    full_prompt = f"""
{TOOL_DESCRIPTION}

User request:
{prompt}
"""

    return client.generate(
        full_prompt,
        temperature=0,
        num_predict=300,
    )

def verify_task(task_spec, tool_result):
    """
    Verify whether the latest tool result provides
    sufficient evidence for task completion.
    """

    if not isinstance(task_spec, dict):
        return {
            "passed": False,
            "reason": "Task specification is invalid.",
            "evidence": {},
        }

    if not isinstance(tool_result, dict):
        return {
            "passed": False,
            "reason": "Tool result is not a dictionary.",
            "evidence": {},
        }

    if tool_result.get("timed_out"):
        return {
            "passed": False,
            "reason": "The command timed out.",
            "evidence": tool_result,
        }

    if tool_result.get("exit_code") != 0:
        return {
            "passed": False,
            "reason": "The command failed.",
            "evidence": tool_result,
        }

    stdout = tool_result.get("stdout", "")

    if not stdout.strip():
        return {
            "passed": False,
            "reason": "The command succeeded but produced no stdout.",
            "evidence": tool_result,
        }

    expected_return = task_spec.get("expected_return")

    if expected_return is not None:
        observed_output = stdout.strip()

        if observed_output != str(expected_return):
            return {
                "passed": False,
                "reason": (
                    f"Expected output {expected_return}, "
                    f"but observed {observed_output!r}."
                ),
                "evidence": {
                    "task_spec": task_spec,
                    "stdout": stdout,
                    "stderr": tool_result.get("stderr", ""),
                    "exit_code": tool_result.get("exit_code"),
                },
            }

    return {
        "passed": True,
        "reason": "The observed output matches the expected return value.",
        "evidence": {
            "task_spec": task_spec,
            "stdout": stdout,
            "stderr": tool_result.get("stderr", ""),
            "exit_code": tool_result.get("exit_code"),
        },
    }

def execute_tool(tool_name, arguments):
    if not isinstance(tool_name, str):
        raise ValueError("Tool name must be a string.")

    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object.")

    tool = TOOLS[tool_name]

    try:
        return tool(**arguments)
    except TypeError as e:
        raise ValueError(
            f"Invalid arguments for tool '{tool_name}': {e}"
        )



def parse_json_response(response):
    response = response.strip()

    # Normal JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # JSON wrapped in Markdown code fences
    if response.startswith("```"):
        lines = response.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Model returned invalid JSON:\n{response}"
    )


def build_prompt(messages):
    parts = []

    for message in messages:
        role = message["role"]
        content = message["content"]

        parts.append(
            f"[{role.upper()}]\n{content}"
        )

    return "\n\n".join(parts)


def validate_final_request(messages, task_spec):
    for message in reversed(messages):
        if message["role"] != "tool":
            continue

        try:
            result = json.loads(message["content"])
        except json.JSONDecodeError:
            continue

        if not isinstance(result, dict):
            continue

        return verify_task(task_spec, result)

    return {
        "passed": False,
        "reason": "No tool result is available to verify.",
        "evidence": {},
    }

def run_agent(user_request, task_spec, max_steps=8):
    messages = [
    {
        "role": "system",
        "content": TOOL_DESCRIPTION,
    },
    {
        "role": "user",
        "content": user_request,
    },
]
    action_history = []
    
    verification_status = {
    "passed": False,
    "reason": "Task has not been verified yet.",
    "evidence": {},
}
    for step in range(max_steps):
        print(f"\n--- Agent step {step + 1} ---")
        prompt = build_prompt(messages)

        response = client.generate(
        prompt,
        temperature=0,
        num_predict=300,
        format="json",
    )
        messages.append(
    {
        "role": "assistant",
        "content": response,
    }
)

        print("Model response:")
        print(response)

        request = parse_json_response(response)

        if not isinstance(request, dict):
            raise ValueError("Model response must be a JSON object.")

        # Agent is finished
        if "final" in request:
            verification_status = validate_final_request(
                    messages,
                    task_spec,
                )

            print("Verification:")
            print(json.dumps(verification_status, indent=2))

            if not verification_status["passed"]:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You cannot finish yet.\n"
                            f"Verification failed: "
                            f"{verification_status['reason']}\n"
                            "Continue using the available tools."
                        ),
                    }
                )
                continue

            return request["final"]

        # Agent wants to use a tool
        if "tool" not in request:
            raise ValueError(
                "Model response must contain either 'tool' or 'final'."
            )

        if "arguments" not in request:
            raise ValueError(
                "Tool request is missing 'arguments'."
            )

        tool_name = request["tool"]
        arguments = request["arguments"]

        key = action_key(
            tool_name,
            arguments,
        )

        current_workspace_state = workspace_fingerprint()

        if action_history:
            previous_action = action_history[-1]

            previous_key = action_key(
                previous_action["tool"],
                previous_action["arguments"],
            )

            if key == previous_key:
                previous_workspace_state = previous_action["workspace_state"]

                if current_workspace_state == previous_workspace_state:
                    result = {
                        "error": "repeated_action",
                        "message": (
                            "This exact action was already attempted with the same "
                            "arguments and the workspace has not changed since then. "
                            "No progress was detected. Inspect the current state and "
                            "choose a different action."
                        ),
                    }

                    messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps(result, indent=2),
                        }
                    )

                    print("Tool result:")
                    print(result)

                    continue


        
        result = execute_tool(
            tool_name,
            arguments,
        )

        action_history.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
                "workspace_state": current_workspace_state,
            }
        )

        messages.append(
    {
        "role": "tool",
        "content": json.dumps(result, indent=2),
    }
    )
        print("Tool result:")
        print(result)

    
    raise RuntimeError(
        f"Agent stopped after reaching the maximum of {max_steps} steps."
    )

user_request = (
    "Create a Python file named loop_test.py inside test-project. "
    "The file must contain a function called calculate that returns 100. "
    "Then verify that calculate() actually returns 100 by executing code "
    "that calls the function and prints its return value. "
    "Do not finish until you have observed 100 in stdout."
)
task_spec = {
    "goal": "calculate() must return 100",
    "file": "test-project/loop_test.py",
    "function": "calculate",
    "expected_return": 100,
}

result = run_agent(
    user_request,
    task_spec,
    max_steps=12,
)

print("\n========== FINAL ANSWER ==========")
print(result)