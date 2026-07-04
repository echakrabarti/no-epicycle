import os
import sys
import json
from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich import print as rprint
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(
    name="noepi",
    help="noepicycle — stop your agentic coding loop from going in circles.",
    add_completion=False,
)
console = Console()


def check_api_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "\n[red]✗ ANTHROPIC_API_KEY not set.[/red]\n\n"
            "Get a key at [link]https://console.anthropic.com[/link]\n"
            "Then either:\n"
            "  [dim]export ANTHROPIC_API_KEY=sk-ant-...[/dim]     (current session)\n"
            "  [dim]echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env[/dim]  (project .env file)\n"
        )
        raise typer.Exit(1)


def check_docker():
    from noepicycle.executor import check_docker_available
    available, message = check_docker_available()
    if not available:
        console.print(f"\n[red]✗ Docker unavailable:[/red] {message}\n")
        raise typer.Exit(1)


def load_test_file(test_file: Path) -> str:
    if not test_file.exists():
        console.print(f"\n[red]✗ Test file not found:[/red] {test_file}\n")
        raise typer.Exit(1)
    return test_file.read_text(encoding="utf-8")


def print_run_header(task: str, budget: int, ladder_name: str, dry_run: bool):
    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]LIVE RUN[/green]"
    console.print(Panel(
        f"[bold]noepicycle[/bold]  {mode}\n\n"
        f"[dim]Task:[/dim]    {task[:80]}{'...' if len(task) > 80 else ''}\n"
        f"[dim]Budget:[/dim]  {budget:,} tokens\n"
        f"[dim]Ladder:[/dim]  {ladder_name}",
        border_style="dim",
    ))


def print_history_table(history):
    table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
    table.add_column("Iter", style="dim", width=5)
    table.add_column("Model", width=24)
    table.add_column("Score", width=8)
    table.add_column("Delta", width=8)
    table.add_column("Tokens", width=8)

    for record in history:
        delta = record["delta"]
        delta_str = f"{delta:+.3f}" if record["iteration"] > 0 else "—"
        delta_color = "green" if delta > 0.02 else ("red" if delta < 0 else "yellow")

        score_pct = f"{record['score']:.0%}"
        score_color = "green" if record["score"] >= 0.8 else ("yellow" if record["score"] >= 0.4 else "red")

        table.add_row(
            str(record["iteration"] + 1),
            record["model"].replace("claude-", "").replace("-20251001", ""),
            f"[{score_color}]{score_pct}[/{score_color}]",
            f"[{delta_color}]{delta_str}[/{delta_color}]",
            f"{record['tokens_used']:,}",
        )

    console.print(table)


def print_result(result, task: str):
    stop_colors = {
        "solved": "green",
        "budget": "yellow",
        "plateau": "yellow",
        "exhausted": "red",
        "fixation": "red",
    }
    stop_color = stop_colors.get(result.stop_reason, "white")

    console.print()
    console.print(Panel(
        f"[bold]Result[/bold]\n\n"
        f"[dim]Stop reason:[/dim]  [{stop_color}]{result.stop_reason}[/{stop_color}]\n"
        f"[dim]Final score:[/dim]  [bold]{result.score:.0%}[/bold] tests passing\n"
        f"[dim]Iterations:[/dim]   {result.iterations}\n"
        f"[dim]Tokens used:[/dim]  {result.tokens_spent:,}\n"
        f"[dim]Models tried:[/dim] {', '.join(m.replace('claude-', '').replace('-20251001', '') for m in result.visited_models)}",
        border_style=stop_color,
    ))

    if result.solution:
        console.print("\n[bold dim]Best solution:[/bold dim]")
        console.print(Panel(result.solution, border_style="dim"))


@app.command()
def run(
    task: str = typer.Argument(..., help="The coding task to solve"),
    test_file: Path = typer.Option(
        ..., "--tests", "-t",
        help="Path to Python test file (functions starting with test_)",
    ),
    budget: int = typer.Option(
        50_000, "--budget", "-b",
        help="Token budget cap",
    ),
    ladder: str = typer.Option(
        "default", "--ladder", "-l",
        help="Ladder to use: 'default' (Claude-only) or 'performance' (Claude+DeepSeek, opt-in)",
    ),
    context_transfer: str = typer.Option(
        "summary", "--context-transfer",
        help="Context transfer mode on model switch: summary | reset | full",
    ),
    success_threshold: float = typer.Option(
        1.0, "--success-threshold",
        help="Score at which to stop (1.0 = all tests passing, 0.8 = 80%)",
    ),
    timeout: int = typer.Option(
        30, "--timeout",
        help="Docker execution timeout per iteration in seconds",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Estimate cost only, do not run",
    ),
    output_json: Optional[Path] = typer.Option(
        None, "--output-json",
        help="Write full results to a JSON file",
    ),
):
    check_api_key()
    check_docker()

    test_code = load_test_file(test_file)

    from noepicycle.ladder import DEFAULT_LADDER, PERFORMANCE_LADDER
    from noepicycle.graph import Supervisor

    if ladder == "performance":
        console.print(
            "\n[yellow]⚠ Performance ladder includes DeepSeek models.[/yellow]\n"
            "[dim]DeepSeek is subject to Chinese data jurisdiction.\n"
            "Do not use with proprietary, sensitive, or regulated code.[/dim]\n"
        )
        ladder_obj = PERFORMANCE_LADDER
        ladder_name = "performance (Claude + DeepSeek)"
    else:
        ladder_obj = DEFAULT_LADDER
        ladder_name = "default (Claude-only)"

    print_run_header(task, budget, ladder_name, dry_run)

    supervisor = Supervisor(
        test_code=test_code,
        budget_cap=budget,
        ladder=ladder_obj,
        context_transfer=context_transfer,
        success_threshold=success_threshold,
        timeout=timeout,
    )

    if dry_run:
        estimate = supervisor.estimate(task)
        console.print("\n[bold]Cost estimate:[/bold]")
        for k, v in estimate.items():
            console.print(f"  [dim]{k}:[/dim] {v}")
        raise typer.Exit(0)

    from noepicycle.executor import pull_docker_image, DOCKER_IMAGE
    with console.status("[dim]Pulling Docker image if needed...[/dim]"):
        pull_docker_image(DOCKER_IMAGE)

    console.print("\n[dim]Running...[/dim]\n")

    try:
        result = supervisor.run(task=task)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(1)

    print_history_table(result.history)
    print_result(result, task)

    if output_json:
        output_data = {
            "task": task,
            "stop_reason": result.stop_reason,
            "score": result.score,
            "iterations": result.iterations,
            "tokens_spent": result.tokens_spent,
            "visited_models": result.visited_models,
            "best_solution": result.solution,
            "history": [
                {
                    "iteration": r["iteration"],
                    "model": r["model"],
                    "score": r["score"],
                    "delta": r["delta"],
                    "tokens_used": r["tokens_used"],
                }
                for r in result.history
            ],
        }
        output_json.write_text(json.dumps(output_data, indent=2))
        console.print(f"\n[dim]Results written to {output_json}[/dim]")


@app.command()
def check():
    console.print("\n[bold]noepicycle environment check[/bold]\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        console.print(f"[green]✓[/green] ANTHROPIC_API_KEY set (sk-ant-...{api_key[-4:]})")
    else:
        console.print("[red]✗[/red] ANTHROPIC_API_KEY not set")

    from noepicycle.executor import check_docker_available
    available, message = check_docker_available()
    if available:
        console.print(f"[green]✓[/green] Docker available")
    else:
        console.print(f"[red]✗[/red] {message}")

    try:
        import langgraph
        try:
            from importlib.metadata import version
            lg_version = version("langgraph")
        except Exception:
            lg_version = "installed"
        console.print(f"[green]✓[/green] langgraph {lg_version}")
    except ImportError:
        console.print("[red]✗[/red] langgraph not installed — run: pip install -e '.[dev]'")

    try:
        import anthropic
        console.print(f"[green]✓[/green] anthropic SDK installed")
    except ImportError:
        console.print("[red]✗[/red] anthropic not installed")

    console.print()


if __name__ == "__main__":
    app()