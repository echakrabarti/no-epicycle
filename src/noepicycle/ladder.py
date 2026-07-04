from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LadderRung:
    model: str
    on_plateau: Optional[str]
    on_regression: Optional[str]
    on_fixation: Optional[str]
    on_budget_low: Optional[str]
    cost_per_1k_output: float = 0.0
    swe_bench_score: float = 0.0
    rank: int = 0


@dataclass
class Ladder:
    rungs: list[LadderRung]

    def __post_init__(self):
        for i, rung in enumerate(self.rungs):
            rung.rank = i
        self._by_model: dict[str, LadderRung] = {r.model: r for r in self.rungs}

    def get(self, model: str) -> Optional[LadderRung]:
        return self._by_model.get(model)

    def next_model(self, current_model, signal, visited_models, tokens_remaining):
        rung = self.get(current_model)
        if rung is None:
            return None
        signal_map = {
            "plateau": rung.on_plateau,
            "regression": rung.on_regression,
            "fixation": rung.on_fixation,
            "budget_low": rung.on_budget_low,
        }
        target = signal_map.get(signal)
        if target is None:
            return None
        if target in visited_models:
            current_rank = rung.rank
            for candidate in self.rungs:
                if (candidate.rank > current_rank
                        and candidate.model not in visited_models
                        and self._can_afford(candidate, tokens_remaining)):
                    return candidate.model
            return None
        target_rung = self.get(target)
        if target_rung and not self._can_afford(target_rung, tokens_remaining):
            return self.best_affordable(tokens_remaining, visited_models)
        return target

    def best_affordable(self, tokens_remaining, visited_models):
        MIN_CYCLES = 3
        affordable = [
            r for r in self.rungs
            if r.model not in visited_models
            and self._can_afford(r, tokens_remaining, MIN_CYCLES)
        ]
        if not affordable:
            return None
        return max(affordable, key=lambda r: r.rank).model

    def _can_afford(self, rung, tokens_remaining, cycles=1):
        TOKENS_PER_CYCLE = 1500
        return tokens_remaining >= (TOKENS_PER_CYCLE * cycles)

    @property
    def starting_model(self):
        return self.rungs[0].model

    @property
    def model_names(self):
        return [r.model for r in self.rungs]


DEFAULT_LADDER = Ladder(rungs=[
    LadderRung(
        model="claude-haiku-4-5-20251001",
        on_plateau="claude-sonnet-4-6",
        on_regression="claude-sonnet-4-6",
        on_fixation="claude-sonnet-4-6",
        on_budget_low=None,
        cost_per_1k_output=0.00125,
        swe_bench_score=0.0,
    ),
    LadderRung(
        model="claude-sonnet-4-6",
        on_plateau="claude-opus-4-8",
        on_regression="claude-opus-4-8",
        on_fixation="claude-opus-4-8",
        on_budget_low="claude-haiku-4-5-20251001",
        cost_per_1k_output=0.015,
        swe_bench_score=0.796,
    ),
    LadderRung(
        model="claude-opus-4-8",
        on_plateau=None,
        on_regression=None,
        on_fixation=None,
        on_budget_low="claude-sonnet-4-6",
        cost_per_1k_output=0.025,
        swe_bench_score=0.886,
    ),
])

PERFORMANCE_LADDER = Ladder(rungs=[
    LadderRung(
        model="claude-haiku-4-5-20251001",
        on_plateau="deepseek-chat",
        on_regression="claude-sonnet-4-6",
        on_fixation="deepseek-chat",
        on_budget_low=None,
        cost_per_1k_output=0.00125,
        swe_bench_score=0.0,
    ),
    LadderRung(
        model="deepseek-chat",
        on_plateau="claude-sonnet-4-6",
        on_regression="claude-sonnet-4-6",
        on_fixation="claude-sonnet-4-6",
        on_budget_low="claude-haiku-4-5-20251001",
        cost_per_1k_output=0.00028,
        swe_bench_score=0.66,
    ),
    LadderRung(
        model="claude-sonnet-4-6",
        on_plateau="deepseek-reasoner",
        on_regression="claude-opus-4-8",
        on_fixation="deepseek-reasoner",
        on_budget_low="deepseek-chat",
        cost_per_1k_output=0.015,
        swe_bench_score=0.796,
    ),
    LadderRung(
        model="deepseek-reasoner",
        on_plateau="claude-opus-4-8",
        on_regression="claude-opus-4-8",
        on_fixation="claude-opus-4-8",
        on_budget_low="claude-sonnet-4-6",
        cost_per_1k_output=0.00219,
        swe_bench_score=0.492,
    ),
    LadderRung(
        model="claude-opus-4-8",
        on_plateau=None,
        on_regression=None,
        on_fixation=None,
        on_budget_low="claude-sonnet-4-6",
        cost_per_1k_output=0.025,
        swe_bench_score=0.886,
    ),
])