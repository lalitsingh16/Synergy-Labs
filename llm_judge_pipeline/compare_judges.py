#!/usr/bin/env python3
"""
compare_judges.py

Runs the evaluation test suite through two different judges:
1. Judge A (same family as the generator model, e.g., OpenAI GPT-4o)
2. Judge B (different family from the generator, e.g., Anthropic Claude 3.5 Sonnet)

Compares their mean scores overall and per-criterion, and counts the number of
disagreements of more than 1.0 point, indicating whether same-family bias is present.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from src.judge import Judge

# Try importing Rich for beautiful terminal output; fall back to stdout if not available
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def print_text_table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    """Print a simple ASCII table as a fallback."""
    if title:
        print(f"\n=== {title} ===")
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
            
    # Format line
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    
    # Print headers
    print(border)
    header_str = "| " + " | ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers)) + " |"
    print(header_str)
    print(border)
    
    # Print rows
    for row in rows:
        row_str = "| " + " | ".join(f"{str(val):<{widths[i]}}" for i, val in enumerate(row)) + " |"
        print(row_str)
    print(border)


def print_rich_table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    """Print a styled Rich table."""
    console = Console()
    table = Table(title=title, show_header=True, header_style="bold magenta", border_style="dim")
    
    for h in headers:
        table.add_column(h)
        
    for row in rows:
        table.add_row(*[str(val) for val in row])
        
    console.print(table)


def main() -> None:
    # 1. Load env and verify path configuration
    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    
    if not env_path.exists():
        print(f"[!] Warning: .env file not found at {env_path}. Copying from .env.example...")
        example_env = project_root / ".env.example"
        if example_env.exists():
            with open(example_env, "r", encoding="utf-8") as src, open(env_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            print("[+] Created .env from .env.example. Please update your API keys before running!")
        else:
            print("[!] Error: .env.example not found either. Please create a .env file.")
            sys.exit(1)

    load_dotenv(dotenv_path=env_path)

    # 2. Get configuration
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    gen_model = os.getenv("GENERATOR_MODEL") or "gpt-4o"
    gen_provider = os.getenv("GENERATOR_PROVIDER") or "openai"
    
    judge_model = os.getenv("JUDGE_MODEL") or "claude-3-5-sonnet-20241022"
    judge_provider = os.getenv("JUDGE_PROVIDER") or "anthropic"

    # Make sure we have keys
    if not openai_key or "your-openai-api-key" in openai_key:
        print("[!] Warning: OpenAI API key is missing or placeholder. OpenAI calls will fail.")
    if not anthropic_key or "your-anthropic-api-key" in anthropic_key:
        print("[!] Warning: Anthropic API key is missing or placeholder. Anthropic calls will fail.")

    test_suite_path = project_root / (os.getenv("TEST_SUITE_PATH") or "data/test_suite.json")
    rubric_path = project_root / (os.getenv("RUBRIC_PATH") or "data/rubric.json")

    # 3. Determine Judge A (same family as generator) vs Judge B (different family)
    # Judge A uses generator model family (usually OpenAI)
    # Judge B uses judge model family (usually Anthropic)
    print(f"[*] Generator Model Family: {gen_provider.upper()} ({gen_model})")
    print(f"[*] Judge Model Family    : {judge_provider.upper()} ({judge_model})")
    print("-" * 60)

    # Instantiate Judge A (Same Family)
    print(f"[*] Initialising Judge A (Same Family: {gen_provider.upper()} / {gen_model})...")
    try:
        judge_a = Judge(
            rubric_path=rubric_path,
            provider=gen_provider,
            model=gen_model,
            log_dir=project_root / "logs"
        )
    except Exception as exc:
        print(f"[!] Error instantiating Judge A: {exc}")
        sys.exit(1)

    # Instantiate Judge B (Different Family)
    print(f"[*] Initialising Judge B (Different Family: {judge_provider.upper()} / {judge_model})...")
    try:
        judge_b = Judge(
            rubric_path=rubric_path,
            provider=judge_provider,
            model=judge_model,
            log_dir=project_root / "logs"
        )
    except Exception as exc:
        print(f"[!] Error instantiating Judge B: {exc}")
        sys.exit(1)

    # 4. Load test suite
    if not test_suite_path.exists():
        print(f"[!] Error: Test suite not found at {test_suite_path}")
        sys.exit(1)

    print(f"[*] Loading test suite from {test_suite_path}...")
    test_cases = Judge.load_test_suite(test_suite_path)
    print(f"[+] Loaded {len(test_cases)} cases successfully.")
    print("-" * 60)

    # 5. Run evaluations
    print("[*] Running evaluations with Judge A (Same Family)...")
    results_a = judge_a.evaluate_batch(test_cases)
    
    print("\n[*] Running evaluations with Judge B (Different Family)...")
    results_b = judge_b.evaluate_batch(test_cases)
    print("-" * 60)

    # 6. Analyze results
    disagreements = []
    scores_a = []
    scores_b = []
    
    # Track criterion scores for each judge
    crit_scores_a = {}
    crit_scores_b = {}

    for r_a, r_b in zip(results_a, results_b):
        if r_a.error or r_b.error:
            continue
            
        scores_a.append(r_a.overall_score)
        scores_b.append(r_b.overall_score)

        # Collect criterion scores
        for c, s in r_a.criterion_scores.items():
            crit_scores_a.setdefault(c, []).append(s)
        for c, s in r_b.criterion_scores.items():
            crit_scores_b.setdefault(c, []).append(s)

        diff = r_a.overall_score - r_b.overall_score
        if abs(diff) > 1.0:
            disagreements.append({
                "id": r_a.test_case_id,
                "score_a": r_a.overall_score,
                "score_b": r_b.overall_score,
                "diff": diff
            })

    # Calculations
    mean_a = sum(scores_a) / len(scores_a) if scores_a else 0.0
    mean_b = sum(scores_b) / len(scores_b) if scores_b else 0.0
    bias_delta = mean_a - mean_b

    # 7. Print overall comparison report
    headers = ["Metric", "Judge A (Same Family)", "Judge B (Different)", "Difference"]
    rows = [
        ["Judge Model", f"{judge_a.provider}/{judge_a.model}", f"{judge_b.provider}/{judge_b.model}", "-"],
        ["Mean Overall Score", f"{mean_a:.2f}", f"{mean_b:.2f}", f"{bias_delta:+.2f}"],
        ["Total Valid Cases Evaluated", str(len(scores_a)), str(len(scores_b)), "-"],
        ["Disagreements (> 1.0 point)", "-", "-", str(len(disagreements))]
    ]
    
    title = "JUDGE COMPARISON SUMMARY (Self-Serving Bias Analysis)"
    if HAS_RICH:
        print_rich_table(headers, rows, title=title)
    else:
        print_text_table(headers, rows, title=title)

    # 8. Print per-criterion mean comparison
    crit_headers = ["Criterion", "Judge A Mean Score", "Judge B Mean Score", "Difference"]
    crit_rows = []
    all_criteria = set(crit_scores_a.keys()).union(crit_scores_b.keys())
    
    for c in sorted(all_criteria):
        m_a = sum(crit_scores_a[c]) / len(crit_scores_a[c]) if c in crit_scores_a else 0.0
        m_b = sum(crit_scores_b[c]) / len(crit_scores_b[c]) if c in crit_scores_b else 0.0
        d = m_a - m_b
        crit_rows.append([c, f"{m_a:.2f}", f"{m_b:.2f}", f"{d:+.2f}"])
        
    crit_title = "PER-CRITERION COMPARISON"
    if HAS_RICH:
        print_rich_table(crit_headers, crit_rows, title=crit_title)
    else:
        print_text_table(crit_headers, crit_rows, title=crit_title)

    # 9. Print notable disagreements
    if disagreements:
        dis_headers = ["Test Case ID", "Judge A Score", "Judge B Score", "Delta (A - B)"]
        dis_rows = []
        for d in disagreements:
            dis_rows.append([d["id"], f"{d['score_a']:.2f}", f"{d['score_b']:.2f}", f"{d['diff']:+.2f}"])
            
        dis_title = "NOTABLE DISAGREEMENTS (> 1.0 point delta)"
        if HAS_RICH:
            print_rich_table(dis_headers, dis_rows, title=dis_title)
        else:
            print_text_table(dis_headers, dis_rows, title=dis_title)
            
        # Analysis statement
        if bias_delta > 0.15:
            analysis = (
                f"[!] Notice: Judge A (same family as generator) scored outputs on average {bias_delta:.2f} points HIGHER "
                "than Judge B. This is consistent with a self-serving / same-family bias."
            )
        elif bias_delta < -0.15:
            analysis = (
                f"[!] Notice: Judge A scored outputs on average {abs(bias_delta):.2f} points LOWER than Judge B. "
                "The same-family judge was more critical."
            )
        else:
            analysis = "[+] Notice: The mean score difference is minor. No strong same-family bias detected."
            
        if HAS_RICH:
            Console().print(Panel(analysis, title="Analysis Note", style="bold yellow"))
        else:
            print(f"\n{analysis}\n")
    else:
        print("\n[+] No notable disagreements (> 1.0 point difference) found between the judges.")


if __name__ == "__main__":
    main()
