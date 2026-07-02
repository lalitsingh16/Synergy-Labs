#!/usr/bin/env python3
"""
test_retest.py

Evaluates LLM-as-judge consistency:
1. Re-runs the judge evaluation 3 times on the same fixed outputs (data/test_suite.json).
2. Computes the % of cases where the verdict (pass/fail or score bucket) was identical.
3. Reports overall consistency scores.
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

    test_suite_path = project_root / (os.getenv("TEST_SUITE_PATH") or "data/test_suite.json")
    rubric_path = project_root / (os.getenv("RUBRIC_PATH") or "data/rubric.json")

    # 2. Instantiate judge
    print(f"[*] Initialising judge model: {judge_provider.upper()} ({judge_model})...")
    try:
        judge = Judge(rubric_path=rubric_path)
    except Exception as exc:
        print(f"[!] Error instantiating Judge: {exc}")
        sys.exit(1)

    # 3. Load test suite
    if not test_suite_path.exists():
        print(f"[!] Error: Test suite not found at {test_suite_path}")
        sys.exit(1)
        
    print(f"[*] Loading test suite from {test_suite_path}...")
    test_cases = Judge.load_test_suite(test_suite_path)
    num_cases = len(test_cases)
    print(f"[+] Loaded {num_cases} test cases.")
    print("-" * 60)

    # 4. Re-run evaluations 3 times
    runs_data = []  # will hold list of JudgeResult lists
    
    for run_num in range(1, 4):
        print(f"[*] Running Judge Evaluation Run {run_num}/3...")
        # We call evaluate_batch or manually run to collect results
        results = judge.evaluate_batch(test_cases)
        runs_data.append(results)
        print(f"[+] Run {run_num}/3 complete.\n")
        
    print("-" * 60)

    # 5. Compute consistency metrics
    pass_fail_consistent_count = 0
    exact_score_consistent_count = 0
    criterion_score_total = 0
    criterion_score_consistent_count = 0

    per_case_metrics = []

    valid_cases_count = 0

    for i in range(num_cases):
        case_id = test_cases[i].get("id", f"case_{i+1}")
        
        # Get overall scores for the 3 runs
        score_1 = runs_data[0][i].overall_score
        score_2 = runs_data[1][i].overall_score
        score_3 = runs_data[2][i].overall_score

        # Check for errors in any runs
        errors = [runs_data[0][i].error, runs_data[1][i].error, runs_data[2][i].error]
        if any(errors):
            print(f"[!] Case {case_id} had errors in some runs: {errors}")
            continue

        valid_cases_count += 1

        # Get pass status for the 3 runs
        pass_1 = judge.did_pass(runs_data[0][i])
        pass_2 = judge.did_pass(runs_data[1][i])
        pass_3 = judge.did_pass(runs_data[2][i])

        # Pass/fail consistency: all three must match
        pass_fail_consistent = (pass_1 == pass_2 == pass_3)
        if pass_fail_consistent:
            pass_fail_consistent_count += 1

        # Exact overall score consistency (within rounding tolerance)
        exact_score_consistent = (abs(score_1 - score_2) < 0.01 and abs(score_2 - score_3) < 0.01)
        if exact_score_consistent:
            exact_score_consistent_count += 1

        # Criterion-level consistency (individual 1-5 scores)
        crit_names = test_cases[i]["criteria"]
        crit_consistent_for_case = 0
        crit_total_for_case = len(crit_names)
        
        for crit in crit_names:
            c_score_1 = runs_data[0][i].criterion_scores.get(crit)
            c_score_2 = runs_data[1][i].criterion_scores.get(crit)
            c_score_3 = runs_data[2][i].criterion_scores.get(crit)
            
            criterion_score_total += 1
            if c_score_1 == c_score_2 == c_score_3 and c_score_1 is not None:
                criterion_score_consistent_count += 1
                crit_consistent_for_case += 1

        crit_consistency = crit_consistent_for_case / crit_total_for_case if crit_total_for_case else 1.0

        per_case_metrics.append({
            "case_id": case_id,
            "scores": [score_1, score_2, score_3],
            "pass_verdicts": [pass_1, pass_2, pass_3],
            "pass_fail_consistent": pass_fail_consistent,
            "exact_score_consistent": exact_score_consistent,
            "criterion_consistency": crit_consistency
        })

    # Consistency percentages
    pass_fail_consistency_pct = (pass_fail_consistent_count / valid_cases_count) * 100.0 if valid_cases_count else 0.0
    exact_score_consistency_pct = (exact_score_consistent_count / valid_cases_count) * 100.0 if valid_cases_count else 0.0
    criterion_consistency_pct = (criterion_score_consistent_count / criterion_score_total) * 100.0 if criterion_score_total else 0.0

    # 6. Print Consistency Report
    headers = ["Metric", "Consistent Cases / Scores", "Total Cases / Scores", "Consistency Score (%)"]
    rows = [
        ["Pass/Fail Verdict Consistency", f"{pass_fail_consistent_count} / {valid_cases_count}", str(valid_cases_count), f"{pass_fail_consistency_pct:.1f}%"],
        ["Exact Overall Score Consistency", f"{exact_score_consistent_count} / {valid_cases_count}", str(valid_cases_count), f"{exact_score_consistency_pct:.1f}%"],
        ["Individual Criterion (1-5) Consistency", f"{criterion_score_consistent_count} / {criterion_score_total}", str(criterion_score_total), f"{criterion_consistency_pct:.1f}%"]
    ]

    title = "LLM-AS-JUDGE TEST-RETEST CONSISTENCY SUMMARY"
    if HAS_RICH:
        print_rich_table(headers, rows, title=title)
    else:
        print_text_table(headers, rows, title=title)

    # 7. Write report to reports/test_retest_report.json
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "test_retest_report.json"

    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "total_test_cases": num_cases,
        "pass_fail_consistency": {
            "consistent_count": pass_fail_consistent_count,
            "total_count": num_cases,
            "percentage": pass_fail_consistency_pct
        },
        "exact_score_consistency": {
            "consistent_count": exact_score_consistent_count,
            "total_count": num_cases,
            "percentage": exact_score_consistency_pct
        },
        "criterion_score_consistency": {
            "consistent_count": criterion_score_consistent_count,
            "total_count": criterion_score_total,
            "percentage": criterion_consistency_pct
        },
        "per_case_metrics": per_case_metrics
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    analysis_text = f"Test-retest report saved to {report_file}"
    if HAS_RICH:
        Console().print(Panel(analysis_text, title="Success", style="bold green"))
    else:
        print(f"\n[+] {analysis_text}\n")


if __name__ == "__main__":
    main()
