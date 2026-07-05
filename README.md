# noepicycle

As [Michael Shimeles](https://www.youtube.com/watch?v=7clJ8IH784Q) has pointed out, unless you work at Claude and have unlimited token access, agentic coding loops are prohibitively expensive. noepicycle is a LangGraph-native supervisor agent that prevents token cost from spiralling by detecting diminishing
returns in real time and dynamically switching models or stopping cleanly
when it's clear you've started burning through the token budget without improvements.

## The problem

Most agentic coding loops run until they hit a fixed iteration cap or
exhaust a token budget. But research shows that **rounds 1-2 capture
75% of reachable improvement** in LLM refinement loops [Williams, 2026].
Much that comes after that is diminishing returns or worse since continuing 
to loop is actively harmful to code quality:

- **Self-conditioning**: models condition on their own prior failures,
  making future mistakes *more* likely, not less [arXiv:2509.09677]
- **Context rot**: each iteration appends more context, degrading
  performance by 15-20 percentage points by the time you've accumulated
  ~4,000 tokens of history [Stanford/Redis, 2025]
- **Oscillation**: loops are more less likely to plateau cleanly than to 
  oscillate, making a single-iteration dip look like recovery when
  it isn't [arXiv:2509.04191]

## How it works

noepicycle runs as an **outer LangGraph supervisor** over your existing
inner coding loop (there's also a bare-bones inner loop provided). 
After each iteration, it:

1. Scores the current solution against your scoring function
2. Computes the improvement delta vs. prior iterations
3. Checks the delta trend against research-backed thresholds
4. Makes one of three decisions:
   - **Continue** — improvement is real, keep going
   - **Switch** — plateau or regression detected; switch to a different
     model with a clean context summary, breaking the self-conditioning
     cycle
   - **Stop** — solved, budget exhausted, or all ladder rungs tried

Model switching follows a **user-configurable ladder** — a ranked
graph of models with explicit transition rules for plateau, regression,
and budget-low events. The default ladder alternates between model
families to break reasoning ruts.

```
[Your inner loop agent]
        ↑  ↓  (one iteration)
[noepicycle supervisor]
  → score → delta → decision
        ↓
  continue / switch model / stop
```

## Benchmark results (v0.1.0)

Evaluated across 5 hard coding tasks (CSV parsing, LRU cache, rate limiting,
expression evaluation, event emitter), 3 runs per condition, compared against
a fixed 5-iteration flat baseline using Claude Haiku.

| Task | noepicycle tokens | flat tokens | savings | accuracy |
|---|---|---|---|---|
| csv_parser | 624 ± 43 | 1,165 | 46% | 100% vs 100% |
| lru_cache | 946 ± 318 | 1,705 | 45% | 100% vs 100% |
| expression_evaluator | 2,034 ± 979 | 2,759 | 26% | **100% vs 96%** |
| event_emitter | 824 ± 53 | 1,362 | 40% | 100% vs 100% |
| hash_ring | 573 ± 225 | 578 | ~0% | 100% vs 100% |

**32% mean token savings across tasks where the supervisor's early-stopping
signal fired. On expression_evaluator, noepicycle also achieved higher
accuracy than the flat baseline (100% vs 96%) at lower cost.**

## Installation

pip install noepicycle

Requires Docker Desktop running (for sandboxed code execution).
Get Docker at https://docker.com/get-started

Set your Anthropic API key:
    export ANTHROPIC_API_KEY=sk-ant-...
    # or create a .env file in your project directory

## Quickstart

**CLI (no setup required beyond API key + Docker):**

    # write a test file
    echo 'from solution import f
def test_basic():
    assert f([1,2,3]) == [3,2,1]' > tests.py

    # run noepicycle
    noepi run "Write a Python function f that reverses a list" \
        --tests tests.py --budget 30000

**Python API:**

    from noepicycle import Supervisor

    supervisor = Supervisor(
        test_code=open("tests.py").read(),
        budget_cap=30_000,
    )
    result = supervisor.run("Write a Python function f that reverses a list")
    print(result.solution)
    print(f"Score: {result.score:.0%}, Tokens: {result.tokens_spent}")

## Plug-and-play interface

noepicycle wraps your loop, not the other way around, although it has a 
canonical inner loop built in. You provide:
1. A **scoring callback** — any function that takes a solution string
   and returns a float (0.0–1.0)
2. A **budget cap** — in tokens
3. Optionally, a **ladder config** — which models to use and when

```python
from noepicycle import Supervisor

supervisor = Supervisor(
    score_fn=lambda solution: run_tests(solution),  # your scorer
    budget_cap=50_000,                               # tokens
    # ladder="default"  # or pass your own
)

result = supervisor.run(
    task="Fix the failing authentication middleware",
    inner_loop=your_langgraph_agent,  # your existing agent
)

print(result.solution)       # best solution found
print(result.iterations)     # how many inner loop iterations ran
print(result.tokens_spent)   # total tokens used
print(result.stop_reason)    # "solved" | "plateau" | "budget" | "exhausted"
```

## The ladder

The supervisor navigates a model ladder — a directed graph of models
with transition rules for different signal types:

```
Signal types:
  on_plateau    → improvement delta < threshold for N consecutive iterations
  on_regression → score decreased from prior iteration (iteration > 0 only)
  on_budget_low → remaining budget below configured threshold

Default ladder (Claude-only, privacy-safe):

  claude-haiku-4-5
      on_plateau    → claude-sonnet-4-6
      on_regression → claude-sonnet-4-6
      on_budget_low → stop

  claude-sonnet-4-6
      on_plateau    → claude-opus-4-8
      on_regression → claude-opus-4-8
      on_budget_low → claude-haiku-4-5

  claude-opus-4-8
      on_plateau    → stop  (terminal)
      on_regression → stop
      on_budget_low → claude-sonnet-4-6
```

Bring your own ladder:

```python
from noepicycle import Supervisor, Ladder

my_ladder = Ladder({
    "claude-haiku-4-5": {
        "on_plateau": "claude-sonnet-4-6",
        "on_regression": "claude-opus-4-8",
        "on_budget_low": None,
    },
    # ... more rungs
})

supervisor = Supervisor(score_fn=..., budget_cap=..., ladder=my_ladder)
```

## Context transfer on model switch

When switching models, noepicycle does not pass the full failure
history to the new model — that would re-introduce the self-conditioning
problem. Instead, it runs a cheap summarization call (haiku) to produce
a clean briefing:

```
Task: [original task]
Best solution so far: [current best]
Approaches tried: [summary of what failed and why]
```

This breaks the self-conditioning cycle while preserving useful signal.
Configurable:

```python
Supervisor(..., context_transfer="summary")  # default
Supervisor(..., context_transfer="reset")    # task + best only, nothing else
Supervisor(..., context_transfer="full")     # full history (not recommended)
```

## Stopping thresholds

Default thresholds are derived from published research:

| Parameter | Default | Source |
|---|---|---|
| delta_threshold | 0.02 | arXiv:2603.27440 |
| plateau_window | 2 | arXiv:2509.06770 |
| budget_low_pct | 0.25 | noepicycle default |
| budget_stop_pct | 0.05 | noepicycle default |
| grace_period | 2 | noepicycle default |

All configurable:

```python
Supervisor(
    ...,
    delta_threshold=0.05,   # stricter plateau detection
    plateau_window=3,        # require more evidence before switching
    grace_period=3,          # more cycles before reassessing after switch
)
```

## Low test coverage warning

noepicycle's convergence detection is most reliable with 5+ objective
test cases. With fewer tests, the scoring signal is coarser and deltas
noisier. If your scorer returns only binary values (0.0 or 1.0),
noepicycle will warn and widen the plateau_window automatically:

```
⚠ noepicycle: Binary scoring signal detected (possible low test
  coverage). Widening plateau_window from 2 to 4 for more reliable
  convergence detection. Consider adding more tests or using
  score_fn="llm_judge" for a smoother signal.
```

## Privacy

noepicycle's default ladder uses Anthropic's Claude models only.
Anthropic's enterprise data handling policies apply.

An optional performance ladder including DeepSeek models is available
but **opt-in only**. DeepSeek is subject to Chinese data jurisdiction
and should not be used with proprietary, sensitive, or regulated code:

```python
# opt-in explicitly — do not use with sensitive codebases
from noepicycle.ladders import PERFORMANCE_LADDER
supervisor = Supervisor(..., ladder=PERFORMANCE_LADDER)
```

See [PRIVACY.md](./PRIVACY.md) for data handling details for each
supported model family.

## Research foundation

noepicycle's design is grounded in published findings:

**On diminishing returns:**
- Williams (2026): Rounds 1-2 capture 75% of reachable improvement
  in LLM refinement loops. [LLM Verification Loops, Medium]
- REA-Coder (2026): Improvement from iterations 1-5 averages 12.18%;
  iterations 5-10 yield only 6% additional gain. [arXiv:2604.16198]
- KubeGuard (2025): Oscillation with diminishing returns emerges
  beyond iteration 3-4; early stopping readily mitigates it.
  [arXiv:2509.04191]

**On self-conditioning (why loops get worse, not just flat):**
- "The Illusion of Diminishing Returns" (2026): Models condition on
  their own prior mistakes, increasing future error rates beyond
  long-context effects alone. [arXiv:2509.09677]
- "Contextual Drag" (2026): Iterative refinement collapses accuracy
  for models with contextual drag; independent samples improve
  steadily. [arXiv:2602.04288]

**On context rot:**
- Stanford/Redis (2025): Accuracy drops 15-20 percentage points with
  ~4,000 tokens of accumulated context. [redis.io/blog/context-rot]

**On coding loops specifically:**
- "Another Turn, Better Output?" (2025): Coding benefits from early
  decision and restraint. If a correct path does not appear quickly,
  stop or restart — do not push vague refinement. [arXiv:2509.06770]

**On default thresholds:**
- delta_threshold=0.02: If two consecutive iterations show delta <
  0.02, additional refinement is unlikely to help. [arXiv:2603.27440]

**On model selection:**
- Harness/scaffolding moves SWE-bench results by 17-21 points — more
  than model swaps alone. noepicycle changes both model and prompting
  strategy on switch. [morphllm.com/best-ai-model-for-coding]
- SWE-bench Verified scores for default ladder: Opus 4.8 (88.6%),
  Sonnet 4.6 (79.6%), Haiku 4.5 (~$0.13/solved task).
  [morphllm.com/best-ai-model-for-coding]

## Status

- [x] README / research foundation
- [x] Core state schema with runtime evidence capture
- [x] Default model ladder (Claude-only + opt-in DeepSeek)
- [x] LangGraph supervisor graph
- [x] Docker-sandboxed executor with intermediate variable tracing
- [x] CLI (noepi run / --dry-run / check)
- [x] Preflight single-shot gate (avoids loop overhead on easy tasks)
- [x] Evaluation harness (3-run benchmark vs flat baseline)
- [x] PyPI package (pip install noepicycle)
- [ ] Loop Library submission
- [ ] MCP server for Claude Code integration
- [ ] Direction injection + constraint extraction (v1.5)
- [ ] Topology learning from run logs
- [ ] Loop-aware inner agent

## Known limitations (v0.1.0)

- High variance on tasks where the model produces slightly different 
  incorrect solutions each iteration — fixation detection requires 
  identical solution hashes, so near-identical broken solutions don't 
  trigger a switch as early as they should. Fix planned for v0.2.0.
- Tested on Claude Haiku only for the default inner loop. Sonnet/Opus 
  as inner loop models not yet benchmarked.
- Windows path handling in Docker executor may require Docker Desktop 
  running in Linux container mode.

## Contributing

Issues and PRs welcome.

If you run noepicycle on a task and get interesting results (especially
cases where the supervisor made a wrong call), open an issue with your
`eval/results.json` — real usage data helps improve the default ladder
and stopping thresholds.

## License

MIT