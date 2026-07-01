from typing import TypedDict, Optional
import hashlib

class TestResult(TypedDict):
    name: str
    passed: bool
    output: str
    # Runtime evidence #
    input_args: Optional[str]
    expected: Optional[str]
    got: Optional [str]
    intermediate_vars: Optional[str]
    stack_trace: Optional[str]
    raw_output: str
    first_seen: int
    fixed_in: Optional[int]
    broken_in: Optional[int]

class IterationRecord(TypedDict):
    iteration: int
    model: str
    score: float # 0-1 depending on how many tests pass
    delta: float #current score - previous--hardcoded for 0 for iteration 0
    tokens_used: int
    solution: str
    solution_hash: str # for fixation detection
    test_results: list[TestResult]

class SupervisorDecision(TypedDict):
    action: str
    reason: str #CLI output
    next_model: Optional[str]

class LoopState(TypedDict):
    task: str
    current_solution: str
    best_score: float
    history: list[IterationRecord]
    budget_cap: int
    tokens_spent: int
    current_model: str
    visited_models: list[str]
    grace_period_remaining: int
    solution_hashes: list[str]
    fixation_count: int
    fixation_threshold: int
    decision: Optional[SupervisorDecision]
    consecutive_plateau_count: int
    delta_threshold: float
    plateau_window: int
    grace_period: int
    context_transfer: str
    success_threshold: float
    stop_reason: Optional[str]

def hash_solution(solution:str) -> str:
    normalized = " ".join(solution.split())
    return hashlib.md5(normalized.encode()).hexdigest()[:16]

def build_debug_thread(history: list[IterationRecord]) -> str:
    if not history:
        return ""
 
    MAX_RAW = 300
    lines = []
 
    # check for fixation across iterations
    hashes = [r["solution_hash"] for r in history]
    repeated_hashes = {h for h in hashes if hashes.count(h) > 1}
 
    for record in history:
        pct = int(record["score"] * 100)
        fixation_flag = ""
        if record["solution_hash"] in repeated_hashes:
            count = hashes[:record["iteration"] + 1].count(record["solution_hash"])
            if count > 1:
                fixation_flag = f"  ⚠ SAME SOLUTION AS ITERATION {hashes.index(record['solution_hash']) + 1} — FIXATION"
 
        lines.append(
            f"\nIteration {record['iteration'] + 1} "
            f"(model: {record['model']}): {pct}% passing"
            f"{fixation_flag}"
        )
 
        for test in record.get("test_results", []):
            if test["passed"]:
                fixed = test.get("fixed_in") == record["iteration"]
                marker = "  (fixed this iteration)" if fixed else ""
                lines.append(f"  ✓ {test['name']}{marker}")
            else:
                regression = test.get("broken_in") == record["iteration"]
                reg_marker = "  ← NEW REGRESSION" if regression else ""
                lines.append(f"  ✗ {test['name']}{reg_marker}")
 
                # runtime evidence — most valuable first
                if test.get("input_args"):
                    lines.append(f"    Input:    {test['input_args']}")
                if test.get("expected"):
                    lines.append(f"    Expected: {test['expected']}")
                if test.get("got"):
                    lines.append(f"    Got:      {test['got']}")
                if test.get("intermediate_vars"):
                    lines.append(f"    Vars:     {test['intermediate_vars']}")
                if test.get("stack_trace"):
                    # indent each line of the stack trace
                    for tline in test["stack_trace"].split("\n"):
                        lines.append(f"    Trace:    {tline}")
                elif test.get("raw_output"):
                    out = test["raw_output"][:MAX_RAW]
                    if len(test["raw_output"]) > MAX_RAW:
                        out += "... [truncated]"
                    lines.append(f"    Error:    {out}")
 
    # add explicit fixation warning at the end if detected
    if repeated_hashes and len(history) > 1:
        lines.append(
            "\n⚠ FIXATION WARNING: You have produced the same solution "
            "multiple times. This approach is not working. "
            "Try a fundamentally different algorithm or data structure."
        )
 
    return "\n".join(lines)

def compute_test_results(
    current_results: dict[str, dict],
    history: list[IterationRecord],
    iteration: int,
) -> list[TestResult]:
    prior_states: dict[str, bool] = {}
    if history:
        for t in history[-1].get("test_results", []):
            prior_states[t["name"]] = t["passed"]
 
    results = []
    for name, data in current_results.items():
        passed = data["passed"]
        was_passing = prior_states.get(name)
 
        fixed_in = iteration if (was_passing is False and passed) else None
        broken_in = iteration if (was_passing is True and not passed) else None
        first_seen = iteration if name not in prior_states else 0
 
        results.append(TestResult(
            name=name,
            passed=passed,
            input_args=data.get("input_args"),
            expected=data.get("expected"),
            got=data.get("got"),
            intermediate_vars=data.get("intermediate_vars"),
            stack_trace=data.get("stack_trace"),
            raw_output=data.get("raw_output", "")[:500],
            first_seen=first_seen,
            fixed_in=fixed_in,
            broken_in=broken_in,
        ))
 
    return results

def initial_state(
    task: str,
    budget_cap: int,
    starting_model: str,
    delta_threshold: float = 0.02,
    plateau_window: int = 2,
    grace_period: int = 2,
    context_transfer: str = "summary",
    success_threshold: float = 1.0,
    fixation_threshold: int = 2,
) -> LoopState:
    return LoopState(
        task=task,
        current_solution="",
        best_solution="",
        best_score=0.0,
        history=[],
        budget_cap=budget_cap,
        tokens_spent=0,
        current_model=starting_model,
        visited_models=[starting_model],
        grace_period_remaining=0,
        solution_hashes=[],
        fixation_count=0,
        fixation_threshold=fixation_threshold,
        decision=None,
        consecutive_plateau_count=0,
        delta_threshold=delta_threshold,
        plateau_window=plateau_window,
        grace_period=grace_period,
        context_transfer=context_transfer,
        success_threshold=success_threshold,
        stop_reason=None,
    )