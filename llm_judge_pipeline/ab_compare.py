#!/usr/bin/env python3
"""
ab_compare.py

Performs an A/B prompt optimization comparison for a generator model:
1. Takes two system prompt configurations (V1 and V2).
2. Generates outputs for each test case in data/test_suite.json under both configurations.
3. Evaluates both generated sets using the configured LLM-as-judge.
4. Aggregates results: mean scores, win rates, and overall winner.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

import openai
import anthropic

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


def generate_response(
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_input: str,
) -> str:
    """Generate a response using the generator model."""
    provider = provider.lower()
    if provider == "openai":
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content or ""
    elif provider == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.7,
            system=system_prompt,
            messages=[{"role": "user", "content": user_input}]
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))
    else:
        raise ValueError(f"Unsupported generator provider '{provider}'")


def main() -> None:
    # 1. Setup paths and load env
    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)

    # 2. Get API credentials & model config
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    gen_model = os.getenv("GENERATOR_MODEL") or "gpt-4o"
    gen_provider = os.getenv("GENERATOR_PROVIDER") or "openai"
    
    judge_model = os.getenv("JUDGE_MODEL") or "claude-3-5-sonnet-20241022"
    judge_provider = os.getenv("JUDGE_PROVIDER") or "anthropic"

    test_suite_path = project_root / (os.getenv("TEST_SUITE_PATH") or "data/test_suite.json")
    rubric_path = project_root / (os.getenv("RUBRIC_PATH") or "data/rubric.json")

    # Verify keys
    gen_key = openai_key if gen_provider == "openai" else anthropic_key
    if not gen_key or "your-" in gen_key:
        print(f"[!] Error: API Key for generator provider '{gen_provider}' is not set in .env.")
        sys.exit(1)

    # 3. Instantiate judge
    print(f"[*] Initialising judge model: {judge_provider.upper()} ({judge_model})...")
    try:
        judge = Judge(rubric_path=rubric_path)
    except Exception as exc:
        print(f"[!] Error instantiating Judge: {exc}")
        sys.exit(1)

    # 4. Load test suite
    if not test_suite_path.exists():
        print(f"[!] Error: Test suite not found at {test_suite_path}")
        sys.exit(1)
        
    print(f"[*] Loading test cases from {test_suite_path}...")
    test_cases = Judge.load_test_suite(test_suite_path)
    print(f"[+] Loaded {len(test_cases)} test cases.")
    print("-" * 60)

    # 5. Define V1 vs V2 Prompt optimization strategies
    # V1: The default system prompt provided in the test case.
    # V2: The system prompt with enhanced detail and quality constraints added.
    nudge = " Answer in a highly structured, thorough manner using clear headers or bullet points. Maintain a highly professional and objective tone."
    
    print("[*] Prompt Optimization Configurations:")
    print("    [V1 System Prompt] -> The original prompt in test_suite.json")
    print(f"    [V2 System Prompt] -> Original prompt + prompt optimization suffix:")
    print(f"                         \"{nudge.strip()}\"")
    print("-" * 60)

    # 6. Generate and evaluate responses for each test case
    detailed_results = []
    
    scores_v1 = []
    scores_v2 = []
    
    crit_scores_v1 = {}
    crit_scores_v2 = {}
    
    v2_wins = 0
    v1_wins = 0
    ties = 0

    print(f"[*] Generating and judging outputs for {len(test_cases)} cases (using {gen_model})...")
    
    for i, case in enumerate(test_cases, 1):
        case_id = case.get("id", f"case_{i}")
        user_input = case["input"]
        criteria = case["criteria"]
        expected = case.get("expected_output")
        
        sys_v1 = case["system_prompt"]
        sys_v2 = sys_v1.rstrip(".") + "." + nudge

        print(f"    [{i}/{len(test_cases)}] Processing {case_id}...")
        
        try:
            # Generate V1 response
            out_v1 = generate_response(gen_provider, gen_model, gen_key, sys_v1, user_input)
            
            # Generate V2 response
            out_v2 = generate_response(gen_provider, gen_model, gen_key, sys_v2, user_input)
        except Exception as exc:
            print(f"    [!] Generation failed for case {case_id}: {exc}")
            continue

        # Evaluate V1
        tc_v1 = {
            "id": f"{case_id}_v1",
            "input": user_input,
            "system_prompt": sys_v1,
            "model_output": out_v1,
            "expected_output": expected,
            "criteria": criteria
        }
        res_v1 = judge.evaluate(tc_v1)

        # Evaluate V2
        tc_v2 = {
            "id": f"{case_id}_v2",
            "input": user_input,
            "system_prompt": sys_v2,
            "model_output": out_v2,
            "expected_output": expected,
            "criteria": criteria
        }
        res_v2 = judge.evaluate(tc_v2)

        if res_v1.error or res_v2.error:
            print(f"    [!] Judging failed for case {case_id} — V1 Error: {res_v1.error}, V2 Error: {res_v2.error}")
            continue

        scores_v1.append(res_v1.overall_score)
        scores_v2.append(res_v2.overall_score)

        # Accumulate per-criterion scores
        for c, s in res_v1.criterion_scores.items():
            crit_scores_v1.setdefault(c, []).append(s)
        for c, s in res_v2.criterion_scores.items():
            crit_scores_v2.setdefault(c, []).append(s)

        # Win rate logic
        if res_v2.overall_score > res_v1.overall_score:
            v2_wins += 1
            comparison = "V2 Won"
        elif res_v1.overall_score > res_v2.overall_score:
            v1_wins += 1
            comparison = "V1 Won"
        else:
            ties += 1
            comparison = "Tie"

        detailed_results.append({
            "case_id": case_id,
            "input": user_input,
            "output_v1": out_v1,
            "output_v2": out_v2,
            "score_v1": res_v1.overall_score,
            "score_v2": res_v2.overall_score,
            "criterion_scores_v1": res_v1.criterion_scores,
            "criterion_scores_v2": res_v2.criterion_scores,
            "comparison": comparison
        })

    # 7. Aggregates
    valid_count = len(scores_v1)
    if valid_count == 0:
        print("[!] Error: No cases successfully processed.")
        sys.exit(1)

    mean_v1 = sum(scores_v1) / valid_count
    mean_v2 = sum(scores_v2) / valid_count
    win_rate = v2_wins / valid_count
    
    if mean_v2 > mean_v1:
        overall_winner = "V2 (Optimised Prompt)"
    elif mean_v1 > mean_v2:
        overall_winner = "V1 (Baseline Prompt)"
    else:
        overall_winner = "Draw"

    # 8. Print aggregate comparison table
    summary_headers = ["Metric", "V1 (Baseline)", "V2 (Optimised)", "Difference / Outcome"]
    summary_rows = [
        ["Generator Model", gen_model, gen_model, "-"],
        ["Mean Overall Score", f"{mean_v1:.2f}", f"{mean_v2:.2f}", f"{mean_v2 - mean_v1:+.2f}"],
        ["Total Test Cases Evaluated", str(valid_count), str(valid_count), "-"],
        ["Head-to-Head Wins", f"{v1_wins} wins", f"{v2_wins} wins", f"{ties} ties"],
        ["Win Rate (V2 > V1)", "-", f"{win_rate:.1%}", "-"],
        ["Overall Winner", "-", "-", overall_winner]
    ]

    title = "A/B PROMPT OPTIMISATION COMPARISON SUMMARY"
    if HAS_RICH:
        print_rich_table(summary_headers, summary_rows, title=title)
    else:
        print_text_table(summary_headers, summary_rows, title=title)

    # 9. Print per-criterion mean comparison
    crit_headers = ["Criterion", "V1 Mean Score", "V2 Mean Score", "Difference"]
    crit_rows = []
    all_criteria = set(crit_scores_v1.keys()).union(crit_scores_v2.keys())
    
    for c in sorted(all_criteria):
        m_v1 = sum(crit_scores_v1[c]) / len(crit_scores_v1[c]) if c in crit_scores_v1 else 0.0
        m_v2 = sum(crit_scores_v2[c]) / len(crit_scores_v2[c]) if c in crit_scores_v2 else 0.0
        diff = m_v2 - m_v1
        crit_rows.append([c, f"{m_v1:.2f}", f"{m_v2:.2f}", f"{diff:+.2f}"])
        
    crit_title = "PER-CRITERION OPTIMISATION COMPARISON"
    if HAS_RICH:
        print_rich_table(crit_headers, crit_rows, title=crit_title)
    else:
        print_text_table(crit_headers, crit_rows, title=crit_title)

    # 10. Write detailed report JSON
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "ab_comparison_report.json"
    
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generator_model": gen_model,
        "judge_model": judge_model,
        "v2_prompt_nudge": nudge,
        "overall_summary": {
            "mean_score_v1": mean_v1,
            "mean_score_v2": mean_v2,
            "v2_win_rate": win_rate,
            "v1_wins": v1_wins,
            "v2_wins": v2_wins,
            "ties": ties,
            "overall_winner": overall_winner
        },
        "per_criterion": {
            c: {
                "mean_v1": sum(crit_scores_v1[c]) / len(crit_scores_v1[c]) if c in crit_scores_v1 else 0.0,
                "mean_v2": sum(crit_scores_v2[c]) / len(crit_scores_v2[c]) if c in crit_scores_v2 else 0.0
            } for c in all_criteria
        },
        "detailed_results": detailed_results
    }
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
        
    analysis_text = f"A/B comparison report saved to {report_file}"
    if HAS_RICH:
        Console().print(Panel(analysis_text, title="Success", style="bold green"))
    else:
        print(f"\n[+] {analysis_text}\n")


if __name__ == "__main__":
    main()
