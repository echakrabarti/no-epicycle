import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
DEFAULT_INNER_MODEL = "claude-haiku-4-5-20251001"


def default_inner_loop(task, previous_solution, previous_feedback, model=DEFAULT_INNER_MODEL):
    if not previous_solution:
        user_message = f"""Write Python code to solve the following task.
Return ONLY the code, no explanation, no markdown fences, no preamble.
The code will be executed directly.

Task:
{task}"""
    else:
        user_message = f"""You are improving a Python solution that has failing tests.

Task:
{task}

Your previous solution:
```python
{previous_solution}
```

What failed:
{previous_feedback if previous_feedback else "Some tests are still failing."}

Write an improved solution. Return ONLY the code, no explanation,
no markdown fences, no preamble. The code will be executed directly."""

    response = _client.messages.create(
        model=model,
        max_tokens=2048,
        system=(
            "You are an expert Python programmer. "
            "When asked for code, return ONLY valid Python code. "
            "No markdown. No explanation. Just the code."
        ),
        messages=[{"role": "user", "content": user_message}],
    )

    solution = response.content[0].text.strip()

    if solution.startswith("```"):
        lines = solution.split("\n")
        solution = "\n".join(lines[1:-1]).strip()

    tokens_used = response.usage.input_tokens + response.usage.output_tokens
    return solution, tokens_used


def build_feedback(test_output, passed, total):
    MAX_FEEDBACK_CHARS = 1500
    header = f"{passed}/{total} tests passing.\n\n"
    if len(test_output) > MAX_FEEDBACK_CHARS:
        truncated = test_output[:MAX_FEEDBACK_CHARS]
        footer = f"\n\n[truncated — {len(test_output) - MAX_FEEDBACK_CHARS} chars omitted]"
        return header + truncated + footer
    return header + test_output