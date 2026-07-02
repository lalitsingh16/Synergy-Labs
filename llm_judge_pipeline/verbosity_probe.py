#!/usr/bin/env python3
"""
verbosity_probe.py

Investigates verbosity bias in the LLM-as-judge pipeline:
1. Takes a single test case (tc_002: Photosynthesis summarisation).
2. Uses the mediocre response from the test suite.
3. Creates a padded, verbose version of the same mediocre response (3x longer, but with no extra factual substance).
4. Evaluates both responses with the LLM-as-judge.
5. Reports whether the score inflated for the verbose response.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

from src.judge import Judge

# Try importing Rich for styled terminal printing
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def print_text_table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    """Fallback text table."""
    if title:
        print(f"\n=== {title} ===")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(border)
    header_str = "| " + " | ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers)) + " |"
    print(header_str)
    print(border)
    for row in rows:
        row_str = "| " + " | ".join(f"{str(val):<{widths[i]}}" for i, val in enumerate(row)) + " |"
        print(row_str)
    print(border)


def print_rich_table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    """Rich styled table."""
    console = Console()
    table = Table(title=title, show_header=True, header_style="bold magenta", border_style="dim")
    for h in headers:
        table.add_column(h)
    for row in rows:
        table.add_row(*[str(val) for val in row])
    console.print(table)


def main() -> None:
    # 1. Setup paths and load env
    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)

    judge_model = os.getenv("JUDGE_MODEL") or "claude-3-5-sonnet-20241022"
    judge_provider = os.getenv("JUDGE_PROVIDER") or "anthropic"

    rubric_path = project_root / (os.getenv("RUBRIC_PATH") or "data/rubric.json")

    # 2. Instantiate judge
    print(f"[*] Initialising judge model: {judge_provider.upper()} ({judge_model})...")
    try:
        judge = Judge(rubric_path=rubric_path)
    except Exception as exc:
        print(f"[!] Error instantiating Judge: {exc}")
        sys.exit(1)

    # 3. Define the Verbosity Probe case (Photosynthesis Summarisation)
    # The two outputs have the exact same semantic content, but V2 is padded to be ~3.5x longer.
    input_text = (
        "Summarize the following passage in one sentence: 'Photosynthesis is the process "
        "by which green plants convert sunlight, water, and carbon dioxide into glucose and "
        "oxygen. It primarily occurs in the chloroplasts of plant cells and is essential "
        "for life on Earth.'"
    )
    system_prompt = (
        "You are a scientific summarizer. Summarize the given passage in exactly one "
        "sentence without adding external information."
    )
    criteria = ["correctness", "faithfulness", "completeness", "instruction_following"]

    mediocre_output = "Photosynthesis is how plants use sunlight to make food, and it's really important for life."
    
    # 3x longer, but introduces zero new information. Just bloated wording.
    verbose_output = (
        "With respect to the biological mechanism of photosynthesis, it is basically the complex "
        "process through which green plants utilize the energy of sunlight in order to convert water "
        "and carbon dioxide into sugar food, and it is also incredibly vital and critical for the "
        "ongoing survival of life on Earth."
    )

    print("-" * 60)
    print("[*] Verbosity Probe Outputs:")
    print(f"    [Mediocre Output (Concise - {len(mediocre_output.split())} words)]:")
    print(f"    \"{mediocre_output}\"")
    print(f"\n    [Verbose Output (Padded - {len(verbose_output.split())} words)]:")
    print(f"    \"{verbose_output}\"")
    print("-" * 60)

    # 4. Evaluate both
    print("[*] Evaluating Mediocre (Concise) output...")
    case_concise = {
        "id": "probe_concise",
        "input": input_text,
        "system_prompt": system_prompt,
        "model_output": mediocre_output,
        "criteria": criteria
    }
    res_concise = judge.evaluate(case_concise)

    print("[*] Evaluating Verbose (Padded) output...")
    case_verbose = {
        "id": "probe_verbose",
        "input": input_text,
        "system_prompt": system_prompt,
        "model_output": verbose_output,
        "criteria": criteria
    }
    res_verbose = judge.evaluate(case_verbose)

    if res_concise.error or res_verbose.error:
        print(f"[!] Evaluation failed. Concise Error: {res_concise.error}, Verbose Error: {res_verbose.error}")
        sys.exit(1)

    # 5. Compare scores
    score_concise = res_concise.overall_score
    score_verbose = res_verbose.overall_score
    delta = score_verbose - score_concise

    # 6. Format metrics table
    headers = ["Criterion", "Concise Score", "Verbose Score", "Delta"]
    rows = []
    
    # Build rows for each evaluated criterion
    for c in criteria:
        sc_c = res_concise.criterion_scores.get(c, 0)
        sc_v = res_verbose.criterion_scores.get(c, 0)
        rows.append([c.capitalize(), str(sc_c), str(sc_v), f"{sc_v - sc_c:+.1f}"])
        
    # Append overall scores row
    rows.append(["OVERALL SCORE", f"{score_concise:.2f}", f"{score_verbose:.2f}", f"{delta:+.2f}"])

    title = "VERBOSITY BIAS PROBE RESULT"
    if HAS_RICH:
        print_rich_table(headers, rows, title=title)
    else:
        print_text_table(headers, rows, title=title)

    # 7. Write report to reports/verbosity_probe_report.json
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "verbosity_probe_report.json"

    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "case_metadata": {
            "input": input_text,
            "system_prompt": system_prompt,
            "criteria": criteria
        },
        "mediocre_concise": {
            "output": mediocre_output,
            "word_count": len(mediocre_output.split()),
            "overall_score": score_concise,
            "criterion_scores": res_concise.criterion_scores,
            "rationale": res_concise.rationale
        },
        "mediocre_verbose": {
            "output": verbose_output,
            "word_count": len(verbose_output.split()),
            "overall_score": score_verbose,
            "criterion_scores": res_verbose.criterion_scores,
            "rationale": res_verbose.rationale
        },
        "delta": delta,
        "verbosity_inflation_detected": delta > 0.25
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # Summary analysis text
    if delta > 0.25:
        analysis = (
            f"[!] Verbosity Bias Detected: The verbose output scored {delta:+.2f} points higher overall, "
            "despite containing exactly the same semantic information. The judge model was swayed by length."
        )
    elif abs(delta) <= 0.25:
        analysis = (
            f"[+] No Verbosity Bias Detected: The scores matched closely (delta: {delta:+.2f}). "
            "The judge successfully differentiated length from quality."
        )
    else:
        analysis = (
            f"[+] Reverse Verbosity Bias: The concise output scored higher by {abs(delta):.2f} points. "
            "The judge favored brevity."
        )

    if HAS_RICH:
        Console().print(Panel(analysis, title="Analysis Conclusion", style="bold yellow"))
    else:
        print(f"\n{analysis}\n")


if __name__ == "__main__":
    main()
