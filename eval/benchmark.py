import os
import sys
import json
import time
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from noepicycle.graph import Supervisor
from noepicycle.executor import Executor, pull_docker_image, DOCKER_IMAGE
from noepicycle.default_loop import default_inner_loop, build_feedback
from noepicycle.ladder import DEFAULT_LADDER


FLAT_BASELINE_ITERATIONS = 5
N_RUNS = 3

TASKS = [
    {
        "name": "csv_parser",
        "difficulty": "hard",
        "supervisor_behavior": "multi_requirement_iteration",
        "prompt": """Write a Python function called parse_csv(text) that parses a CSV string
and returns a list of lists. Handle: quoted fields containing commas, empty fields,
and trailing newlines. Do not use the csv module.""",
        "test_code": """
from solution import parse_csv

def test_simple():
    assert parse_csv("a,b,c\\n1,2,3") == [["a","b","c"],["1","2","3"]]

def test_quoted_comma():
    assert parse_csv('"hello, world",b') == [["hello, world","b"]]

def test_empty_field():
    assert parse_csv("a,,c") == [["a","","c"]]

def test_trailing_newline():
    assert parse_csv("a,b\\n") == [["a","b"]]

def test_single_field():
    assert parse_csv("hello") == [["hello"]]
""",
    },

    {
        "name": "lru_cache",
        "difficulty": "hard",
        "supervisor_behavior": "fixation_architecture_switch",
        "prompt": """Write a Python class called LRUCache with an __init__(self, capacity) method
and two methods: get(key) returning the value or -1 if not found, and put(key, value)
inserting or updating. When capacity is exceeded, evict the least recently used item.
Both get and put must be O(1). Do not use OrderedDict.""",
        "test_code": """
from solution import LRUCache

def test_basic_put_get():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1

def test_eviction():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.put(3, 3)
    assert c.get(1) == -1

def test_update_recency():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.get(1)
    c.put(3, 3)
    assert c.get(2) == -1
    assert c.get(1) == 1

def test_update_value():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(1, 10)
    assert c.get(1) == 10

def test_capacity_one():
    c = LRUCache(1)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == -1
    assert c.get(2) == 2

def test_missing_key():
    c = LRUCache(2)
    assert c.get(99) == -1
""",
    },

    {
        "name": "rate_limiter",
        "difficulty": "hard",
        "supervisor_behavior": "iterate_on_timing_logic",
        "prompt": """Write a Python class called RateLimiter with __init__(self, max_calls, period_seconds).
Method is_allowed(timestamp) returns True if a call at this timestamp is within the rate limit
(max_calls per period_seconds sliding window), False otherwise. Timestamps are floats in seconds.""",
        "test_code": """
from solution import RateLimiter

def test_basic_allow():
    r = RateLimiter(3, 1.0)
    assert r.is_allowed(0.0) == True
    assert r.is_allowed(0.3) == True
    assert r.is_allowed(0.6) == True

def test_basic_deny():
    r = RateLimiter(3, 1.0)
    r.is_allowed(0.0)
    r.is_allowed(0.3)
    r.is_allowed(0.6)
    assert r.is_allowed(0.9) == False

def test_sliding_window():
    r = RateLimiter(3, 1.0)
    r.is_allowed(0.0)
    r.is_allowed(0.3)
    r.is_allowed(0.6)
    assert r.is_allowed(1.1) == True

def test_exact_boundary():
    r = RateLimiter(2, 1.0)
    r.is_allowed(0.0)
    r.is_allowed(0.5)
    assert r.is_allowed(1.0) == True

def test_single_call_limit():
    r = RateLimiter(1, 1.0)
    assert r.is_allowed(0.0) == True
    assert r.is_allowed(0.5) == False
    assert r.is_allowed(1.1) == True
""",
    },

    {
        "name": "expression_evaluator",
        "difficulty": "hard",
        "supervisor_behavior": "multi_step_improvement",
        "prompt": """Write a Python function called evaluate(expr) that evaluates a math
expression string containing integers, +, -, *, /, and parentheses.
Respect standard operator precedence. / is integer division (truncate toward zero).
No eval() or compile() allowed.""",
        "test_code": """
from solution import evaluate

def test_addition():
    assert evaluate("1+2") == 3

def test_precedence():
    assert evaluate("2+3*4") == 14

def test_parens():
    assert evaluate("(2+3)*4") == 20

def test_nested_parens():
    assert evaluate("((2+3)*4)+1") == 21

def test_subtraction():
    assert evaluate("10-3-2") == 5

def test_division():
    assert evaluate("10/3") == 3

def test_complex():
    assert evaluate("2*(3+4)-1") == 13

def test_spaces():
    assert evaluate("2 + 3 * 4") == 14
""",
    },

    {
        "name": "mini_interpreter",
        "difficulty": "hard",
        "supervisor_behavior": "architecture_requiring_multiple_passes",
        "prompt": """Write a Python function called run(program) that interprets a simple language.
The program is a list of strings, one instruction per line.
Instructions:
  SET x 5       (assign variable x = 5)
  ADD x y z     (x = y + z, where y and z can be variable names or integers)
  PRINT x       (append value of variable x to output)
  IF x GOTO n   (if x != 0, jump to line n, 1-indexed)
Return a list of printed values as integers.""",
        "test_code": """
from solution import run

def test_set_print():
    assert run(["SET x 5", "PRINT x"]) == [5]

def test_add():
    assert run(["SET x 3", "SET y 4", "ADD z x y", "PRINT z"]) == [7]

def test_add_literal():
    assert run(["SET x 3", "ADD y x 7", "PRINT y"]) == [10]

def test_multiple_prints():
    assert run(["SET x 1", "SET y 2", "PRINT x", "PRINT y"]) == [1, 2]

def test_goto():
    assert run(["SET x 0", "SET y 5", "IF y GOTO 5", "PRINT x", "SET x 1", "PRINT x"]) == [1]

def test_loop():
    result = run([
        "SET i 3",
        "SET sum 0",
        "ADD sum sum i",
        "ADD i i -1",
        "IF i GOTO 3",
        "PRINT sum",
    ])
    assert result == [6]
""",
    },
]


def run_flat_baseline(task, executor, n_iterations=FLAT_BASELINE_ITERATIONS):
    total_tokens = 0
    solution = ""
    feedback = ""
    score = 0.0
    history = []

    for i in range(n_iterations):
        sol, tokens = default_inner_loop(
            task=task["prompt"],
            previous_solution=solution,
            previous_feedback=feedback,
            model="claude-haiku-4-5-20251001",
        )
        total_tokens += tokens
        result = executor.run(sol, iteration=i, history=[])
        score = result.score
        solution = sol
        passed = result.passed_count
        total = result.total_count
        feedback = build_feedback(
            "\n".join(
                f"{name}: {'PASS' if r['passed'] else 'FAIL - ' + r['raw_output'][:100]}"
                for name, r in result.test_results.items()
            ),
            passed, total,
        )
        history.append({"iteration": i, "score": score, "tokens": tokens})
        if score >= 1.0:
            break

    return {
        "score": score,
        "tokens_spent": total_tokens,
        "iterations": len(history),
    }


def run_single_shot(task, executor):
    sol, tokens = default_inner_loop(
        task=task["prompt"],
        previous_solution="",
        previous_feedback="",
        model="claude-haiku-4-5-20251001",
    )
    result = executor.run(sol, iteration=0, history=[])
    return {
        "score": result.score,
        "tokens_spent": tokens,
    }


def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


def stdev(lst):
    if len(lst) < 2:
        return 0.0
    return statistics.stdev(lst)


@dataclass
class TaskResult:
    task_name: str
    difficulty: str
    supervisor_behavior: str
    n_runs: int

    noepi_scores: list
    noepi_tokens: list
    noepi_iterations: list
    noepi_stop_reasons: list

    flat_scores: list
    flat_tokens: list
    flat_iterations: list

    single_scores: list
    single_tokens: list

    @property
    def noepi_score_mean(self): return avg(self.noepi_scores)
    @property
    def noepi_score_std(self): return stdev(self.noepi_scores)
    @property
    def noepi_tokens_mean(self): return avg(self.noepi_tokens)
    @property
    def noepi_tokens_std(self): return stdev(self.noepi_tokens)

    @property
    def flat_score_mean(self): return avg(self.flat_scores)
    @property
    def flat_tokens_mean(self): return avg(self.flat_tokens)

    @property
    def single_score_mean(self): return avg(self.single_scores)
    @property
    def single_tokens_mean(self): return avg(self.single_tokens)

    @property
    def tokens_saved_mean(self): return self.flat_tokens_mean - self.noepi_tokens_mean
    @property
    def pct_saved_mean(self):
        if self.flat_tokens_mean == 0:
            return 0.0
        return (self.tokens_saved_mean / self.flat_tokens_mean) * 100

    @property
    def noepi_wins(self):
        return self.tokens_saved_mean > 0 and self.noepi_score_mean >= self.flat_score_mean


def run_benchmark(budget=25000, n_runs=N_RUNS):
    print("\n" + "="*60)
    print("noepicycle benchmark")
    print("="*60)
    print(f"Tasks: {len(TASKS)}")
    print(f"Runs per task per condition: {n_runs}")
    print(f"Budget per task: {budget:,} tokens")
    print(f"Flat baseline iterations: {FLAT_BASELINE_ITERATIONS}")
    print("="*60 + "\n")

    print("Pulling Docker image...")
    pull_docker_image(DOCKER_IMAGE)
    print("Docker ready.\n")

    results = []

    for task in TASKS:
        print(f"-- {task['name']} ({task['difficulty']}) --")
        executor = Executor(test_code=task["test_code"])

        noepi_scores, noepi_tokens_list, noepi_iters, noepi_stops = [], [], [], []
        flat_scores, flat_tokens_list, flat_iters = [], [], []
        single_scores, single_tokens_list = [], []

        for run_i in range(n_runs):
            print(f"  run {run_i + 1}/{n_runs}")

            print(f"    [1/3] noepicycle...")
            supervisor = Supervisor(
                test_code=task["test_code"],
                budget_cap=budget,
                ladder=DEFAULT_LADDER,
                success_threshold=1.0,
            )
            nr = supervisor.run(task=task["prompt"])
            noepi_scores.append(nr.score)
            noepi_tokens_list.append(nr.tokens_spent)
            noepi_iters.append(nr.iterations)
            noepi_stops.append(nr.stop_reason)
            print(f"         score={nr.score:.0%} tokens={nr.tokens_spent} stop={nr.stop_reason}")

            print(f"    [2/3] flat baseline...")
            fr = run_flat_baseline(task, executor)
            flat_scores.append(fr["score"])
            flat_tokens_list.append(fr["tokens_spent"])
            flat_iters.append(fr["iterations"])
            print(f"         score={fr['score']:.0%} tokens={fr['tokens_spent']} iters={fr['iterations']}")

            print(f"    [3/3] single shot...")
            sr = run_single_shot(task, executor)
            single_scores.append(sr["score"])
            single_tokens_list.append(sr["tokens_spent"])
            print(f"         score={sr['score']:.0%} tokens={sr['tokens_spent']}")

        result = TaskResult(
            task_name=task["name"],
            difficulty=task["difficulty"],
            supervisor_behavior=task["supervisor_behavior"],
            n_runs=n_runs,
            noepi_scores=noepi_scores,
            noepi_tokens=noepi_tokens_list,
            noepi_iterations=noepi_iters,
            noepi_stop_reasons=noepi_stops,
            flat_scores=flat_scores,
            flat_tokens=flat_tokens_list,
            flat_iterations=flat_iters,
            single_scores=single_scores,
            single_tokens=single_tokens_list,
        )
        results.append(result)

        saved = result.tokens_saved_mean
        saved_str = f"+{saved:.0f}" if saved > 0 else f"{saved:.0f}"
        print(f"  avg saved: {saved_str} tokens ({result.pct_saved_mean:.0f}%) vs flat baseline")
        print(f"  noepi: {result.noepi_tokens_mean:.0f} +/- {result.noepi_tokens_std:.0f} tokens, score {result.noepi_score_mean:.0%}")
        print(f"  flat:  {result.flat_tokens_mean:.0f} tokens, score {result.flat_score_mean:.0%}\n")

    print_summary(results, n_runs)
    save_results(results)
    return results


def print_summary(results, n_runs):
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_noepi = sum(r.noepi_tokens_mean for r in results)
    total_flat = sum(r.flat_tokens_mean for r in results)
    total_saved = total_flat - total_noepi
    pct_saved = total_saved / total_flat * 100 if total_flat > 0 else 0

    avg_noepi_score = avg([r.noepi_score_mean for r in results])
    avg_flat_score = avg([r.flat_score_mean for r in results])

    wins = [r for r in results if r.noepi_wins]
    losses = [r for r in results if not r.noepi_wins]

    print(f"\nRuns per condition: {n_runs} (means reported)")
    print(f"Total tokens -- noepicycle: {total_noepi:.0f}  flat: {total_flat:.0f}")
    print(f"Net tokens saved: {total_saved:.0f} ({pct_saved:.1f}% vs flat baseline)")
    print(f"Avg score -- noepicycle: {avg_noepi_score:.0%}  flat: {avg_flat_score:.0%}")
    print(f"Task outcomes: {len(wins)} noepicycle wins, {len(losses)} flat wins")

    print(f"\nPer-task breakdown (means over {n_runs} runs):")
    print(f"  {'Task':<25} {'noepi':>9} {'flat':>7} {'saved':>8} {'%saved':>7} {'noepi':>7} {'flat':>6}")
    print(f"  {'':25} {'tok+/-std':>9} {'tokens':>7} {'tokens':>8} {'':>7} {'score':>7} {'score':>6}")
    print("  " + "-"*78)

    for r in results:
        saved_str = f"+{r.tokens_saved_mean:.0f}" if r.tokens_saved_mean > 0 else f"{r.tokens_saved_mean:.0f}"
        win_marker = " <" if r.noepi_wins else "  "
        noepi_str = f"{r.noepi_tokens_mean:.0f}+/-{r.noepi_tokens_std:.0f}"
        print(
            f"  {r.task_name:<25} "
            f"{noepi_str:>9} "
            f"{r.flat_tokens_mean:>7.0f} "
            f"{saved_str:>8} "
            f"{r.pct_saved_mean:>6.0f}% "
            f"{r.noepi_score_mean:>7.0%} "
            f"{r.flat_score_mean:>6.0%}"
            f"{win_marker}"
        )

    if wins:
        avg_savings_on_wins = avg([r.pct_saved_mean for r in wins])
    else:
        avg_savings_on_wins = 0.0

    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    print(f"Across {n_runs} runs per condition on {len(results)} hard coding tasks,")
    print(f"noepicycle matched or exceeded flat baseline accuracy on {len(wins)}/{len(results)} tasks.")
    if wins:
        print(f"On tasks where noepicycle outperformed the flat baseline,")
        print(f"it reduced token cost by {avg_savings_on_wins:.0f}% on average (mean over {n_runs} runs).")
    print(f"Overall mean token delta vs flat {FLAT_BASELINE_ITERATIONS}-iteration baseline: {pct_saved:.0f}%")
    if losses:
        print(f"\nTasks where flat baseline won ({len(losses)}):")
        for r in losses:
            print(f"  {r.task_name}: noepi {r.noepi_tokens_mean:.0f} vs flat {r.flat_tokens_mean:.0f} tokens "
                  f"(flat solved in {avg(r.flat_iterations):.1f} iters avg — supervisor overhead not offset)")
    print(f"{'='*60}\n")


def save_results(results):
    out_path = Path(__file__).parent / "results.json"
    data = []
    for r in results:
        d = {
            "task_name": r.task_name,
            "difficulty": r.difficulty,
            "supervisor_behavior": r.supervisor_behavior,
            "n_runs": r.n_runs,
            "noepi": {
                "score_mean": round(r.noepi_score_mean, 4),
                "score_std": round(r.noepi_score_std, 4),
                "tokens_mean": round(r.noepi_tokens_mean, 1),
                "tokens_std": round(r.noepi_tokens_std, 1),
                "iterations_mean": round(avg(r.noepi_iterations), 2),
                "stop_reasons": r.noepi_stop_reasons,
                "raw_scores": r.noepi_scores,
                "raw_tokens": r.noepi_tokens,
            },
            "flat": {
                "score_mean": round(r.flat_score_mean, 4),
                "tokens_mean": round(r.flat_tokens_mean, 1),
                "iterations_mean": round(avg(r.flat_iterations), 2),
                "raw_scores": r.flat_scores,
                "raw_tokens": r.flat_tokens,
            },
            "single_shot": {
                "score_mean": round(r.single_score_mean, 4),
                "tokens_mean": round(r.single_tokens_mean, 1),
                "raw_scores": r.single_scores,
                "raw_tokens": r.single_tokens,
            },
            "tokens_saved_mean": round(r.tokens_saved_mean, 1),
            "pct_saved_mean": round(r.pct_saved_mean, 2),
            "noepi_wins": r.noepi_wins,
        }
        data.append(d)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Full results written to {out_path}")



SPIRAL_TASKS = [
    {
        "name": "regex_engine",
        "difficulty": "extreme",
        "supervisor_behavior": "plateau_detection_on_hard_task",
        "prompt": """Write a Python function called regex_match(pattern, text) that
returns True if pattern matches the entire text. Support: . (any char),
* (zero or more of preceding), + (one or more of preceding), ? (zero or one of preceding),
[] (character class e.g. [abc]), ^ (start anchor), $ (end anchor).
Do not use the re module.""",
        "test_code": """
from solution import regex_match

def test_literal():
    assert regex_match("abc", "abc") == True

def test_no_match():
    assert regex_match("abc", "xyz") == False

def test_dot():
    assert regex_match("a.c", "abc") == True

def test_star_zero():
    assert regex_match("a*b", "b") == True

def test_star_many():
    assert regex_match("a*b", "aaab") == True

def test_plus_one():
    assert regex_match("a+b", "ab") == True

def test_plus_zero_fails():
    assert regex_match("a+b", "b") == False

def test_anchor_start():
    assert regex_match("^abc", "abc") == True

def test_anchor_full():
    assert regex_match("^abc$", "xabc") == False
""",
    },

    {
        "name": "event_emitter",
        "difficulty": "extreme",
        "supervisor_behavior": "multi_component_spiral",
        "prompt": """Write a Python class called EventEmitter with these methods:
on(event, callback) — subscribe callback to event
off(event, callback) — unsubscribe callback from event
once(event, callback) — subscribe callback that fires only once then auto-unsubscribes
emit(event, *args) — call all callbacks subscribed to event with args
Callbacks receive the args passed to emit.""",
        "test_code": """
from solution import EventEmitter

def test_basic_emit():
    e = EventEmitter()
    results = []
    e.on("data", lambda x: results.append(x))
    e.emit("data", 42)
    assert results == [42]

def test_multiple_listeners():
    e = EventEmitter()
    r1, r2 = [], []
    e.on("x", lambda v: r1.append(v))
    e.on("x", lambda v: r2.append(v))
    e.emit("x", 1)
    assert r1 == [1] and r2 == [1]

def test_off():
    e = EventEmitter()
    results = []
    fn = lambda v: results.append(v)
    e.on("x", fn)
    e.off("x", fn)
    e.emit("x", 1)
    assert results == []

def test_once():
    e = EventEmitter()
    results = []
    e.once("x", lambda v: results.append(v))
    e.emit("x", 1)
    e.emit("x", 2)
    assert results == [1]

def test_no_listeners():
    e = EventEmitter()
    e.emit("nothing", 1)

def test_multiple_args():
    e = EventEmitter()
    results = []
    e.on("x", lambda a, b: results.append((a, b)))
    e.emit("x", 1, 2)
    assert results == [(1, 2)]
""",
    },

    {
        "name": "hash_ring",
        "difficulty": "extreme",
        "supervisor_behavior": "fixation_on_wrong_invariant",
        "prompt": """Write a Python class called HashRing implementing consistent hashing.
__init__(self) creates an empty ring.
add_node(node) adds a node (string) to the ring.
remove_node(node) removes a node from the ring.
get_node(key) returns the node responsible for key (string).
Each node maps to one point on a 0-359 degree ring using hash(node) % 360.
get_node returns the next node clockwise from hash(key) % 360.
If no nodes exist, raises ValueError.""",
        "test_code": """
from solution import HashRing

def test_single_node():
    r = HashRing()
    r.add_node("a")
    assert r.get_node("x") == "a"
    assert r.get_node("y") == "a"

def test_empty_raises():
    r = HashRing()
    try:
        r.get_node("x")
        assert False
    except ValueError:
        pass

def test_add_remove():
    r = HashRing()
    r.add_node("a")
    r.add_node("b")
    r.remove_node("a")
    assert r.get_node("x") == "b"

def test_deterministic():
    r = HashRing()
    r.add_node("server1")
    r.add_node("server2")
    assert r.get_node("mykey") == r.get_node("mykey")

def test_valid_node_returned():
    r = HashRing()
    r.add_node("node1")
    r.add_node("node2")
    for key in ["test1", "test2", "test3", "key1", "key2"]:
        assert r.get_node(key) in ["node1", "node2"]
""",
    },
]

TASKS.extend(SPIRAL_TASKS)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=25000)
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()

    if args.tasks:
        filtered = [t for t in TASKS if t["name"] in args.tasks]
        if not filtered:
            print(f"No tasks found matching: {args.tasks}")
            sys.exit(1)
        TASKS.clear()
        TASKS.extend(filtered)

    run_benchmark(budget=args.budget, n_runs=args.runs)

# These tasks are appended at runtime — add to TASKS list below