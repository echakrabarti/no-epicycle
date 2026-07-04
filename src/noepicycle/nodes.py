import os
from typing import Any, Callable, Optional
from anthropic import Anthropic
from dotenv import load_dotenv

from noepicycle.state import (
    LoopState, IterationRecord, SupervisorDecision,
    build_debug_thread, compute_test_results, hash_solution,
)
from noepicycle.ladder import Ladder, DEFAULT_LADDER
from noepicycle.default_loop import default_inner_loop, build_feedback
from noepicycle.executor import Executor, ExecutionResult

load_dotenv()
_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SUPERVISOR_MODEL = "claude-haiku-4-5-20251001"
SUMMARIZER_MODEL = "claude-haiku-4-5-20251001"


def make_inner_loop_node(executor: Executor, inner_loop_fn: Optional[Callable] = None):
    def inner_loop_node(state: dict) -> dict:
        model = state.get("current_model", "claude-haiku-4-5-20251001")
        history = state.get("history", [])
        previous_solution = state.get("best_solution", "")
        debug_thread = build_debug_thread(history)
        previous_feedback = debug_thread if debug_thread else ""

        if inner_loop_fn is not None:
            solution, tokens_used = inner_loop_fn(
                task=state["task"],
                previous_solution=previous_solution,
                previous_feedback=previous_feedback,
                model=model,
            )
        else:
            solution, tokens_used = default_inner_loop(
                task=state["task"],
                previous_solution=previous_solution,
                previous_feedback=previous_feedback,
                model=model,
            )

        return {
            "current_solution": solution,
            "tokens_spent": state.get("tokens_spent", 0) + tokens_used,
        }
    return inner_loop_node


def make_score_node(executor: Executor):
    def score_node(state: dict) -> dict:
        history = state.get("history", [])
        iteration = len(history)
        solution = state.get("current_solution", "")
        solution_h = hash_solution(solution)

        exec_result: ExecutionResult = executor.run(
            solution_code=solution,
            iteration=iteration,
            history=history,
        )

        score = exec_result.score
        prev_score = history[-1]["score"] if history else 0.0
        delta = score - prev_score if iteration > 0 else 0.0

        test_results = compute_test_results(
            current_results=exec_result.test_results,
            history=history,
            iteration=iteration,
        )

        record = IterationRecord(
            iteration=iteration,
            model=state.get("current_model", ""),
            score=score,
            delta=delta,
            tokens_used=exec_result.tokens_used,
            solution=solution,
            solution_hash=solution_h,
            test_results=test_results,
        )

        new_best_solution = state.get("best_solution", "")
        new_best_score = state.get("best_score", 0.0)
        if score > new_best_score:
            new_best_solution = solution
            new_best_score = score

        existing_hashes = state.get("solution_hashes", [])
        fixation_count = state.get("fixation_count", 0)
        if solution_h in existing_hashes:
            fixation_count += 1
        else:
            fixation_count = 0

        consecutive_plateau_count = state.get("consecutive_plateau_count", 0)
        delta_threshold = state.get("delta_threshold", 0.02)
        if iteration > 0 and abs(delta) < delta_threshold:
            consecutive_plateau_count += 1
        else:
            consecutive_plateau_count = 0

        grace_period_remaining = max(0, state.get("grace_period_remaining", 0) - 1)

        return {
            "history": [record],
            "best_solution": new_best_solution,
            "best_score": new_best_score,
            "solution_hashes": [solution_h],
            "fixation_count": fixation_count,
            "consecutive_plateau_count": consecutive_plateau_count,
            "grace_period_remaining": grace_period_remaining,
            "tokens_spent": state.get("tokens_spent", 0) + exec_result.tokens_used,
        }
    return score_node


def make_supervisor_node(ladder: Ladder):
    def supervisor_node(state: dict) -> dict:
        history = state.get("history", [])
        score = history[-1]["score"] if history else 0.0
        iteration = len(history)
        budget_cap = state.get("budget_cap", 50000)
        tokens_spent = state.get("tokens_spent", 0)
        tokens_remaining = budget_cap - tokens_spent
        grace = state.get("grace_period_remaining", 0)
        success_threshold = state.get("success_threshold", 1.0)

        if score >= success_threshold:
            return {
                "decision": SupervisorDecision(
                    action="stop",
                    reason=f"Solved: score {score:.2f} >= threshold {success_threshold}",
                    next_model=None,
                ),
                "stop_reason": "solved",
            }

        budget_pct = tokens_remaining / budget_cap if budget_cap > 0 else 0
        if budget_pct <= 0.05:
            return {
                "decision": SupervisorDecision(
                    action="stop",
                    reason=f"Budget exhausted: {tokens_spent}/{budget_cap} tokens used",
                    next_model=None,
                ),
                "stop_reason": "budget",
            }

        if grace > 0:
            return {
                "decision": SupervisorDecision(
                    action="continue",
                    reason=f"Grace period: {grace} iterations remaining for new model",
                    next_model=None,
                ),
            }

        signal = None
        reason = ""
        delta_threshold = state.get("delta_threshold", 0.02)
        plateau_window = state.get("plateau_window", 2)
        fixation_threshold = state.get("fixation_threshold", 2)
        fixation_count = state.get("fixation_count", 0)
        consecutive_plateau_count = state.get("consecutive_plateau_count", 0)

        if iteration > 0:
            delta = history[-1]["delta"]
            if delta < 0:
                signal = "regression"
                reason = f"Regression: score dropped {abs(delta):.3f} from prior iteration"
            elif fixation_count >= fixation_threshold:
                signal = "fixation"
                reason = f"Fixation: same solution repeated {fixation_count} times"
            elif consecutive_plateau_count >= plateau_window:
                signal = "plateau"
                reason = f"Plateau: delta < {delta_threshold} for {consecutive_plateau_count} consecutive iterations"

        if budget_pct <= 0.10:
            signal = "budget_low"
            reason = f"Budget low: {int(budget_pct * 100)}% remaining"

        if signal is None:
            last_delta = history[-1]["delta"] if history else 0.0
            return {
                "decision": SupervisorDecision(
                    action="continue",
                    reason=f"Iteration {iteration}: score={score:.2f}, delta={last_delta:.3f}",
                    next_model=None,
                ),
            }

        visited_models = state.get("visited_models", [])
        current_model = state.get("current_model", ladder.starting_model)

        next_model = ladder.next_model(
            current_model=current_model,
            signal=signal,
            visited_models=visited_models,
            tokens_remaining=tokens_remaining,
        )

        if next_model is None:
            model_names = ladder.model_names
            all_tried = all(m in visited_models for m in model_names)
            stop_reason = "exhausted" if all_tried else "plateau"
            return {
                "decision": SupervisorDecision(
                    action="stop",
                    reason=f"{reason} — no viable next model available",
                    next_model=None,
                ),
                "stop_reason": stop_reason,
            }

        return {
            "decision": SupervisorDecision(
                action="switch",
                reason=reason,
                next_model=next_model,
            ),
        }
    return supervisor_node


def make_switch_node():
    def switch_node(state: dict) -> dict:
        decision = state.get("decision", {})
        next_model = decision.get("next_model", "")
        context_transfer = state.get("context_transfer", "summary")
        history = state.get("history", [])
        grace_period = state.get("grace_period", 2)
        tokens_spent = state.get("tokens_spent", 0)
        summary_tokens = 0

        if context_transfer == "summary" and history:
            best_score = state.get("best_score", 0.0)
            n_iters = len(history)
            approaches_tried = list(dict.fromkeys(r["model"] for r in history))
            failing_tests = [
                t["name"] for t in history[-1].get("test_results", [])
                if not t["passed"]
            ]
            summary_prompt = f"""Summarize this coding session for a new AI model taking over. Be concise.

Task: {state.get("task", "")}

Best solution so far (score {best_score:.0%}):
{state.get("best_solution", "")[:800]}

After {n_iters} iterations with {approaches_tried}, still failing: {failing_tests}

Write 2-3 sentences: what the task requires, best approach so far, what's still failing. Do not reproduce the full solution."""

            response = _client.messages.create(
                model=SUMMARIZER_MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": summary_prompt}],
            )
            summary_tokens = response.usage.input_tokens + response.usage.output_tokens

        return {
            "current_model": next_model,
            "visited_models": [next_model],
            "grace_period_remaining": grace_period,
            "consecutive_plateau_count": 0,
            "fixation_count": 0,
            "tokens_spent": tokens_spent + summary_tokens,
        }
    return switch_node


def route(state: dict) -> str:
    decision = state.get("decision")
    if decision is None:
        return "inner_loop"
    action = decision.get("action", "continue")
    if action == "continue":
        return "inner_loop"
    elif action == "switch":
        return "switch"
    elif action == "stop":
        return "end"
    return "inner_loop"