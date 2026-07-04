import os
import json
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

from noepicycle.state import TestResult, compute_test_results, IterationRecord, hash_solution

DOCKER_IMAGE = "python:3.11-slim"
DEFAULT_TIMEOUT_SECONDS = 30
MEMORY_LIMIT = "256m"
CPU_LIMIT = "1.0"

HARNESS_TEMPLATE = '''
import sys
import json
import traceback
import inspect
import re
from typing import Any

MAX_TRACE_EVENTS = 5
MAX_VAR_LEN = 200

class VariableTracer:
    def __init__(self, target_func_name: str):
        self.target = target_func_name
        self.snapshots = []
        self.event_count = 0

    def trace(self, frame, event, arg):
        if frame.f_code.co_name != self.target:
            return self.trace
        if event not in ("line", "return", "exception"):
            return self.trace
        if self.event_count >= MAX_TRACE_EVENTS:
            return None
        local_vars = {}
        for k, v in frame.f_locals.items():
            if k.startswith("_"):
                continue
            try:
                val_str = repr(v)[:MAX_VAR_LEN]
            except Exception:
                val_str = "<unrepresentable>"
            local_vars[k] = val_str
        if local_vars:
            self.snapshots.append({"event": event, "line": frame.f_lineno, "locals": local_vars})
            self.event_count += 1
        return self.trace

    def summary(self):
        if not self.snapshots:
            return None
        parts = []
        for snap in self.snapshots:
            vars_str = ", ".join(f"{k}={v}" for k, v in snap["locals"].items())
            parts.append(f"line {snap['line']} ({snap['event']}): {vars_str}")
        return " | ".join(parts)


def strip_library_frames(tb_str: str) -> str:
    lines = tb_str.strip().split("\\n")
    filtered = []
    skip_next = False
    for line in lines:
        if line.startswith("Traceback"):
            filtered.append(line)
            continue
        if any(x in line for x in ["site-packages", "/usr/lib", "/usr/local/lib", "harness.py", "<string>", "importlib"]):
            skip_next = True
            continue
        if skip_next and line.startswith("    "):
            skip_next = False
            continue
        skip_next = False
        filtered.append(line)
    return "\\n".join(filtered)[:800]


def try_extract_assertion_values(test_source: str, actual_exception: str):
    match = re.search(r"assert\\s+(.+?)\\s*==\\s*(.+?)(?:\\s*,|$)", test_source)
    if not match:
        return None, None
    expected_expr = match.group(2).strip()
    try:
        expected_val = repr(eval(expected_expr))
    except Exception:
        expected_val = expected_expr
    got_val = None
    if "AssertionError:" in actual_exception:
        got_val = actual_exception.split("AssertionError:")[-1].strip()[:200]
    return expected_val, got_val


results = {}

try:
    import solution as sol
except Exception as e:
    import_error = traceback.format_exc()
    for test_name in TEST_NAMES:
        results[test_name] = {
            "passed": False,
            "raw_output": f"Import error: {import_error[:500]}",
            "input_args": None, "expected": None,
            "got": None, "intermediate_vars": None,
            "stack_trace": strip_library_frames(import_error),
        }
    print(json.dumps(results))
    sys.exit(0)

import tests as test_module

for test_name in TEST_NAMES:
    test_func = getattr(test_module, test_name, None)
    if test_func is None:
        results[test_name] = {
            "passed": False,
            "raw_output": f"Test function {test_name} not found",
            "input_args": None, "expected": None,
            "got": None, "intermediate_vars": None, "stack_trace": None,
        }
        continue

    try:
        test_source = inspect.getsource(test_func)
    except Exception:
        test_source = ""

    func_under_test = None
    for name in dir(sol):
        if not name.startswith("_") and callable(getattr(sol, name)):
            if name in test_source:
                func_under_test = name
                break

    tracer = VariableTracer(func_under_test) if func_under_test else None

    try:
        if tracer:
            sys.settrace(tracer.trace)
        test_func()
        if tracer:
            sys.settrace(None)
        results[test_name] = {
            "passed": True,
            "raw_output": "",
            "input_args": None, "expected": None, "got": None,
            "intermediate_vars": tracer.summary() if tracer else None,
            "stack_trace": None,
        }
    except AssertionError as e:
        if tracer:
            sys.settrace(None)
        tb_str = traceback.format_exc()
        expected, got = try_extract_assertion_values(test_source, str(e))
        results[test_name] = {
            "passed": False,
            "raw_output": str(e)[:300] or "AssertionError (no message)",
            "input_args": None, "expected": expected, "got": got,
            "intermediate_vars": tracer.summary() if tracer else None,
            "stack_trace": strip_library_frames(tb_str),
        }
    except Exception as e:
        if tracer:
            sys.settrace(None)
        tb_str = traceback.format_exc()
        results[test_name] = {
            "passed": False,
            "raw_output": f"{type(e).__name__}: {str(e)[:300]}",
            "input_args": None, "expected": None, "got": None,
            "intermediate_vars": tracer.summary() if tracer else None,
            "stack_trace": strip_library_frames(tb_str),
        }

print(json.dumps(results))
'''


class ExecutionResult:
    def __init__(self, test_results, score, tokens_used, error=None, execution_time=0.0):
        self.test_results = test_results
        self.score = score
        self.tokens_used = tokens_used
        self.error = error
        self.execution_time = execution_time

    @property
    def passed_count(self):
        return sum(1 for r in self.test_results.values() if r["passed"])

    @property
    def total_count(self):
        return len(self.test_results)


class Executor:
    def __init__(self, test_code, timeout=DEFAULT_TIMEOUT_SECONDS,
                 memory_limit=MEMORY_LIMIT, cpu_limit=CPU_LIMIT,
                 docker_image=DOCKER_IMAGE):
        self.test_code = test_code
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.docker_image = docker_image
        self.test_names = self._extract_test_names(test_code)
        if not self.test_names:
            raise ValueError("No test functions found. Test functions must start with 'test_'.")
        self._binary_signal_warning = len(self.test_names) < 5

    def run(self, solution_code, iteration, history):
        start_time = time.time()
        if self._binary_signal_warning:
            import warnings
            warnings.warn(
                f"\n⚠ noepicycle: Low test coverage ({len(self.test_names)} tests). "
                f"Convergence detection less reliable with fewer than 5 tests.",
                stacklevel=2,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "solution.py").write_text(solution_code, encoding="utf-8")
            (tmp / "tests.py").write_text(self.test_code, encoding="utf-8")
            test_names_literal = json.dumps(self.test_names)
            harness_code = f"TEST_NAMES = {test_names_literal}\n" + HARNESS_TEMPLATE
            (tmp / "harness.py").write_text(harness_code, encoding="utf-8")

            # fix path for Windows/Git Bash
            mount_path = str(tmp).replace("\\", "/")
            if len(mount_path) > 2 and mount_path[1] == ":":
                mount_path = "/" + mount_path[0].lower() + mount_path[2:]

            cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", self.memory_limit,
                "--cpus", self.cpu_limit,
                "--read-only",
                "--tmpfs", "/tmp:size=64m",
                "-v", f"{mount_path}:/code:ro",
                "-w", "/code",
                self.docker_image,
                "python", "harness.py",
            ]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
                execution_time = time.time() - start_time

                if proc.returncode != 0 and not proc.stdout.strip():
                    return ExecutionResult(
                        test_results=self._all_failed(f"Docker error: {proc.stderr[:500]}"),
                        score=0.0, tokens_used=0,
                        error=proc.stderr[:500], execution_time=execution_time,
                    )

                try:
                    raw_results = json.loads(proc.stdout.strip())
                except json.JSONDecodeError:
                    error_msg = (proc.stdout + proc.stderr)[:500]
                    return ExecutionResult(
                        test_results=self._all_failed(f"Harness crashed: {error_msg}"),
                        score=0.0, tokens_used=0,
                        error=error_msg, execution_time=execution_time,
                    )

                passed = sum(1 for r in raw_results.values() if r["passed"])
                total = len(raw_results)
                score = passed / total if total > 0 else 0.0
                return ExecutionResult(
                    test_results=raw_results, score=score,
                    tokens_used=0, execution_time=execution_time,
                )

            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                error_msg = f"Timed out after {self.timeout}s. Solution may contain an infinite loop."
                return ExecutionResult(
                    test_results=self._all_failed(error_msg),
                    score=0.0, tokens_used=0,
                    error=error_msg, execution_time=execution_time,
                )

    def _extract_test_names(self, test_code):
        import re
        return re.findall(r"^def (test_\w+)", test_code, re.MULTILINE)

    def _all_failed(self, error_msg):
        return {
            name: {
                "passed": False, "raw_output": error_msg,
                "input_args": None, "expected": None,
                "got": None, "intermediate_vars": None, "stack_trace": None,
            }
            for name in self.test_names
        }


def pull_docker_image(image=DOCKER_IMAGE):
    try:
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_docker_available():
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return False, "Docker binary found but returned an error."
    except FileNotFoundError:
        return False, "Docker not found. Install from https://docker.com/get-started"
    except subprocess.TimeoutExpired:
        return False, "Docker version check timed out."
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return False, "Docker installed but daemon isn't running. Start Docker Desktop and try again."
    except subprocess.TimeoutExpired:
        return False, "Docker daemon check timed out. Is Docker Desktop running?"
    return True, "Docker available."