from typing import Optional, Callable
from langgraph.graph import StateGraph, END

from noepicycle.state import LoopState, initial_state
from noepicycle.ladder import Ladder, DEFAULT_LADDER
from noepicycle.executor import Executor
from noepicycle.nodes import (
    make_preflight_node,
    make_inner_loop_node,
    make_score_node,
    make_supervisor_node,
    make_switch_node,
    preflight_route,
    route,
)


def build_graph(
    executor: Executor,
    ladder: Ladder = DEFAULT_LADDER,
    inner_loop_fn: Optional[Callable] = None,
):
    preflight_node = make_preflight_node(executor, inner_loop_fn)
    inner_loop_node = make_inner_loop_node(executor, inner_loop_fn)
    score_node = make_score_node(executor)
    supervisor_node = make_supervisor_node(ladder)
    switch_node = make_switch_node()

    graph = StateGraph(LoopState)

    graph.add_node("preflight", preflight_node)
    graph.add_node("inner_loop", inner_loop_node)
    graph.add_node("score", score_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("switch", switch_node)

    graph.set_entry_point("preflight")

    graph.add_conditional_edges(
        "preflight",
        preflight_route,
        {
            "end": END,
            "supervisor": "supervisor",
        },
    )

    graph.add_edge("inner_loop", "score")
    graph.add_edge("score", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "inner_loop": "inner_loop",
            "switch": "switch",
            "end": END,
        },
    )

    graph.add_edge("switch", "inner_loop")

    return graph.compile()


class RunResult:
    def __init__(self, state: LoopState):
        self.solution = state["best_solution"]
        self.score = state["best_score"]
        self.iterations = len(state["history"])
        self.tokens_spent = state["tokens_spent"]
        self.stop_reason = state["stop_reason"]
        self.history = state["history"]
        self.visited_models = state["visited_models"]

    def __repr__(self):
        return (
            f"RunResult(score={self.score:.2%}, iterations={self.iterations}, "
            f"tokens={self.tokens_spent}, stop={self.stop_reason})"
        )


class Supervisor:
    def __init__(
        self,
        score_fn=None,
        test_code: Optional[str] = None,
        budget_cap: int = 50_000,
        ladder: Ladder = DEFAULT_LADDER,
        inner_loop_fn: Optional[Callable] = None,
        delta_threshold: float = 0.02,
        plateau_window: int = 2,
        grace_period: int = 2,
        context_transfer: str = "summary",
        success_threshold: float = 1.0,
        fixation_threshold: int = 2,
        timeout: int = 30,
    ):
        if test_code is None and score_fn is not None:
            raise ValueError(
                "score_fn is not yet supported directly — pass test_code instead. "
                "score_fn support coming in v1.5."
            )
        if test_code is None:
            raise ValueError("test_code is required. Pass your test suite as a Python string.")

        self.executor = Executor(test_code=test_code, timeout=timeout)
        self.ladder = ladder
        self.inner_loop_fn = inner_loop_fn
        self.budget_cap = budget_cap
        self.delta_threshold = delta_threshold
        self.plateau_window = plateau_window
        self.grace_period = grace_period
        self.context_transfer = context_transfer
        self.success_threshold = success_threshold
        self.fixation_threshold = fixation_threshold

        self._graph = build_graph(
            executor=self.executor,
            ladder=ladder,
            inner_loop_fn=inner_loop_fn,
        )

    def run(self, task: str) -> RunResult:
        state = initial_state(
            task=task,
            budget_cap=self.budget_cap,
            starting_model=self.ladder.starting_model,
            delta_threshold=self.delta_threshold,
            plateau_window=self.plateau_window,
            grace_period=self.grace_period,
            context_transfer=self.context_transfer,
            success_threshold=self.success_threshold,
            fixation_threshold=self.fixation_threshold,
        )
        final_state = self._graph.invoke(state)
        return RunResult(final_state)

    def estimate(self, task: str) -> dict:
        estimated_iterations = 4
        model_cost_per_1k = 0.00125
        estimated_tokens_per_iter = 2000
        estimated_total = estimated_iterations * estimated_tokens_per_iter
        estimated_cost = (estimated_total / 1000) * model_cost_per_1k
        return {
            "estimated_iterations": estimated_iterations,
            "estimated_tokens": estimated_total,
            "estimated_cost_usd": round(estimated_cost, 4),
            "budget_cap": self.budget_cap,
            "budget_pct_used": round(estimated_total / self.budget_cap * 100, 1),
            "starting_model": self.ladder.starting_model,
            "note": "Estimate is approximate. Actual cost depends on task complexity.",
        }