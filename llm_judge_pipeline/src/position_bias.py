"""
position_bias.py — Pairwise position-bias detection for LLM judges.

Tests whether a judge's preference between two model outputs changes when
their presentation order is swapped (A, B) → (B, A).  A high flip rate
indicates the judge is sensitive to *position* rather than *quality*.

Usage:
    from src.judge import Judge
    from src.position_bias import PositionBiasDetector

    judge = Judge(
        rubric_path="data/rubric.json",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key="sk-ant-...",
    )

    detector = PositionBiasDetector(judge)

    # Single pair
    result = detector.compare_pair(pair_case)

    # Full suite
    summary = detector.run_bias_suite(pair_cases)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.judge import (
    Judge,
    JudgeResult,
    _format_rubric_section,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pairwise system prompt
# ---------------------------------------------------------------------------

_PAIRWISE_SYSTEM_PROMPT = """\
You are an expert LLM evaluator performing a **pairwise comparison**. You will \
be shown two model responses (Response A and Response B) to the same user \
input and system prompt.

## Rules — follow these exactly:

1. **Evaluate both responses** against each criterion listed in the rubric.
2. **Pick a winner** for each criterion independently: "A", "B", or "tie".
3. **Pick an overall winner**: "A", "B", or "tie", based on weighted criteria \
importance.
4. **Ground every choice** with a one-sentence justification citing *specific \
evidence* from the responses. Do NOT give generic praise.
5. **Ignore presentation order.** Focus only on content quality.
6. **Return ONLY valid JSON** — no markdown fences, no extra commentary.

## Required JSON schema:

```
{
  "criterion_winners": { "<criterion_name>": "A" | "B" | "tie", ... },
  "criterion_rationale": { "<criterion_name>": "<one-sentence justification>", ... },
  "overall_winner": "A" | "B" | "tie",
  "overall_rationale": "<one-sentence summary of why the winner is better overall>"
}
```
"""


# ---------------------------------------------------------------------------
# Pairwise prompt builder
# ---------------------------------------------------------------------------

def _format_pairwise_request(
    *,
    user_input: str,
    system_prompt: str,
    response_a: str,
    response_b: str,
    rubric_section: str,
    expected_output: str | None = None,
) -> str:
    """Build the user-turn message for a pairwise comparison."""
    parts: list[str] = [
        "Compare the following two model responses.\n",
        "---",
        f"**System prompt given to both models:**\n{system_prompt}\n",
        f"**User input:**\n{user_input}\n",
        f"**Response A:**\n{response_a}\n",
        f"**Response B:**\n{response_b}\n",
    ]

    if expected_output:
        parts.append(
            f"**Reference / expected output (use as a guide, not a rigid template):**"
            f"\n{expected_output}\n"
        )

    parts.append("---\n")
    parts.append(rubric_section)
    parts.append(
        "\nCompare both responses and return ONLY the JSON object described above."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

Winner = Literal["A", "B", "tie"]


@dataclass
class PairwiseVerdict:
    """Result of a single pairwise comparison in one presentation order."""

    order_label: str                                    # "AB" or "BA"
    overall_winner: Winner = "tie"
    criterion_winners: dict[str, Winner] = field(default_factory=dict)
    criterion_rationale: dict[str, str] = field(default_factory=dict)
    overall_rationale: str = ""
    raw_response: str = ""
    error: str | None = None


@dataclass
class PairComparisonResult:
    """Combined result of running both orderings for one pair."""

    pair_id: str
    verdict_ab: PairwiseVerdict = field(default_factory=lambda: PairwiseVerdict(order_label="AB"))
    verdict_ba: PairwiseVerdict = field(default_factory=lambda: PairwiseVerdict(order_label="BA"))
    flipped: bool = False
    winner_ab: Winner = "tie"              # winner in *original* identity space
    winner_ba: Winner = "tie"              # winner in *original* identity space


@dataclass
class BiasSummary:
    """Suite-level position-bias statistics."""

    total_pairs: int = 0
    flip_count: int = 0
    flip_rate: float = 0.0
    consistent_count: int = 0
    flip_examples: list[dict[str, Any]] = field(default_factory=list)
    per_pair: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Position Bias Detector
# ---------------------------------------------------------------------------

class PositionBiasDetector:
    """
    Detects position bias in an LLM judge by running pairwise comparisons
    in both orderings (A, B) and (B, A), then checking for flips.

    Parameters
    ----------
    judge : Judge
        An initialised ``Judge`` instance — its ``_call_llm()`` method and
        rubric are reused for the pairwise calls.
    """

    def __init__(self, judge: Judge) -> None:
        self.judge = judge

    # ------------------------------------------------------------------
    # Core: single pairwise call
    # ------------------------------------------------------------------

    def _judge_pair(
        self,
        *,
        pair_id: str,
        user_input: str,
        system_prompt: str,
        response_a: str,
        response_b: str,
        criteria: list[str],
        expected_output: str | None = None,
        order_label: str = "AB",
    ) -> PairwiseVerdict:
        """
        Run one pairwise comparison through the judge LLM.

        Uses ``Judge._call_llm()`` with the pairwise system prompt so we
        get the same provider / model / temperature configuration.
        """
        from src.judge import parse_verdict, _JUDGE_SYSTEM_PROMPT  # noqa: F811

        rubric_section = _format_rubric_section(self.judge.rubric, criteria)

        user_message = _format_pairwise_request(
            user_input=user_input,
            system_prompt=system_prompt,
            response_a=response_a,
            response_b=response_b,
            rubric_section=rubric_section,
            expected_output=expected_output,
        )

        messages = [{"role": "user", "content": user_message}]

        try:
            llm_resp = self.judge._call_llm(
                messages,
                system_prompt=_PAIRWISE_SYSTEM_PROMPT,
            )
        except Exception as exc:
            logger.error(
                "Pairwise judge call failed for %s (%s): %s",
                pair_id, order_label, exc,
            )
            return PairwiseVerdict(
                order_label=order_label,
                error=f"API call failed: {exc}",
            )

        # Parse the JSON verdict.
        try:
            parsed = parse_verdict(llm_resp.text)
        except ValueError as exc:
            logger.error(
                "JSON parse failed for %s (%s): %s",
                pair_id, order_label, exc,
            )
            return PairwiseVerdict(
                order_label=order_label,
                raw_response=llm_resp.text,
                error=f"JSON parse failed: {exc}",
            )

        # Log the call via the judge's JSONL logger.
        self.judge._log_call(
            test_case_id=f"{pair_id}|pairwise|{order_label}",
            judge_prompt=user_message,
            raw_response=llm_resp.text,
            parsed_verdict=parsed,
            token_usage=llm_resp.usage,
            attempt=1,
        )

        return PairwiseVerdict(
            order_label=order_label,
            overall_winner=parsed.get("overall_winner", "tie"),
            criterion_winners=parsed.get("criterion_winners", {}),
            criterion_rationale=parsed.get("criterion_rationale", {}),
            overall_rationale=parsed.get("overall_rationale", ""),
            raw_response=llm_resp.text,
        )

    # ------------------------------------------------------------------
    # Public: compare one pair in both orderings
    # ------------------------------------------------------------------

    def compare_pair(self, pair_case: dict[str, Any]) -> PairComparisonResult:
        """
        Run a pairwise comparison in both orderings and detect a flip.

        Parameters
        ----------
        pair_case : dict
            Required keys:
              - ``id``: unique pair identifier
              - ``input``: the user query
              - ``system_prompt``: system prompt given to both models
              - ``output_a``: first model's response
              - ``output_b``: second model's response
              - ``criteria``: list of criterion names to evaluate
            Optional:
              - ``expected_output``

        Returns
        -------
        PairComparisonResult
        """
        pair_id = pair_case["id"]
        user_input = pair_case["input"]
        sys_prompt = pair_case["system_prompt"]
        output_a = pair_case["output_a"]
        output_b = pair_case["output_b"]
        criteria = pair_case["criteria"]
        expected = pair_case.get("expected_output")

        logger.info("Position bias check: %s — order (A, B)...", pair_id)

        # --- Order 1: (A, B) —————————————————————————————————————————
        verdict_ab = self._judge_pair(
            pair_id=pair_id,
            user_input=user_input,
            system_prompt=sys_prompt,
            response_a=output_a,
            response_b=output_b,
            criteria=criteria,
            expected_output=expected,
            order_label="AB",
        )

        logger.info("Position bias check: %s — order (B, A)...", pair_id)

        # --- Order 2: (B, A) —————————————————————————————————————————
        verdict_ba = self._judge_pair(
            pair_id=pair_id,
            user_input=user_input,
            system_prompt=sys_prompt,
            response_a=output_b,         # swap
            response_b=output_a,         # swap
            criteria=criteria,
            expected_output=expected,
            order_label="BA",
        )

        # --- Translate winners back to original identity space --------
        #
        #   In the AB call, "A" means output_a won.
        #   In the BA call, "A" means output_b won (since it was placed in
        #   position A). So to map back: swap A↔B in the BA verdict.

        winner_ab = verdict_ab.overall_winner

        ba_raw = verdict_ba.overall_winner
        if ba_raw == "A":
            winner_ba = "B"      # output_b was in position A, so B wins
        elif ba_raw == "B":
            winner_ba = "A"      # output_a was in position B, so A wins
        else:
            winner_ba = "tie"

        flipped = winner_ab != winner_ba

        if flipped:
            logger.warning(
                "FLIP detected for %s: AB→%s, BA→%s",
                pair_id, winner_ab, winner_ba,
            )

        return PairComparisonResult(
            pair_id=pair_id,
            verdict_ab=verdict_ab,
            verdict_ba=verdict_ba,
            flipped=flipped,
            winner_ab=winner_ab,
            winner_ba=winner_ba,
        )

    # ------------------------------------------------------------------
    # Public: run full bias suite
    # ------------------------------------------------------------------

    def run_bias_suite(
        self,
        pair_cases: list[dict[str, Any]],
        report_dir: str | Path = "reports",
        max_flip_examples: int = 3,
    ) -> BiasSummary:
        """
        Run position-bias detection across multiple pair cases.

        Parameters
        ----------
        pair_cases : list[dict]
            List of pair case dicts (same schema as ``compare_pair``).
        report_dir : str | Path
            Directory to write ``position_bias_report.json``.
        max_flip_examples : int
            Number of flip examples to include in the summary (default 3).

        Returns
        -------
        BiasSummary
            Aggregated flip statistics and examples.
        """
        results: list[PairComparisonResult] = []
        flip_examples: list[dict[str, Any]] = []

        for i, pc in enumerate(pair_cases, 1):
            pair_id = pc.get("id", f"pair_{i}")
            logger.info(
                "Bias suite: pair %s (%d/%d)", pair_id, i, len(pair_cases)
            )
            result = self.compare_pair(pc)
            results.append(result)

            if result.flipped and len(flip_examples) < max_flip_examples:
                flip_examples.append({
                    "pair_id": result.pair_id,
                    "input_preview": pc["input"][:120],
                    "winner_when_AB": result.winner_ab,
                    "winner_when_BA": result.winner_ba,
                    "rationale_AB": result.verdict_ab.overall_rationale,
                    "rationale_BA": result.verdict_ba.overall_rationale,
                })

        # --- Compute summary stats ----------------------------------------
        total = len(results)
        flip_count = sum(1 for r in results if r.flipped)
        flip_rate = round(flip_count / total, 4) if total else 0.0
        consistent = total - flip_count

        summary = BiasSummary(
            total_pairs=total,
            flip_count=flip_count,
            flip_rate=flip_rate,
            consistent_count=consistent,
            flip_examples=flip_examples,
            per_pair=[
                {
                    "pair_id": r.pair_id,
                    "winner_AB": r.winner_ab,
                    "winner_BA": r.winner_ba,
                    "flipped": r.flipped,
                }
                for r in results
            ],
        )

        # --- Print human-readable summary ---------------------------------
        self._print_summary(summary)

        # --- Write report to disk -----------------------------------------
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        out_file = report_path / "position_bias_report.json"

        report_dict = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "judge_provider": self.judge.provider,
            "judge_model": self.judge.model,
            "total_pairs": summary.total_pairs,
            "flip_count": summary.flip_count,
            "flip_rate": summary.flip_rate,
            "consistent_count": summary.consistent_count,
            "flip_examples": summary.flip_examples,
            "per_pair": summary.per_pair,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        logger.info("Position bias report written to %s", out_file)

        return summary

    # ------------------------------------------------------------------
    # Pretty-print summary
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(summary: BiasSummary) -> None:
        """Print a human-readable position-bias summary to stdout."""
        border = "=" * 60
        print(f"\n{border}")
        print("  POSITION BIAS REPORT")
        print(border)
        print(f"  Total pairs evaluated : {summary.total_pairs}")
        print(f"  Consistent judgements : {summary.consistent_count}")
        print(f"  Flipped judgements    : {summary.flip_count}")
        print(f"  Flip rate             : {summary.flip_rate:.1%}")
        print(border)

        if summary.flip_examples:
            print("\n  FLIP EXAMPLES (winner changed when order was swapped):")
            print("  " + "-" * 56)
            for ex in summary.flip_examples:
                print(f"\n  Pair: {ex['pair_id']}")
                print(f"    Input : {ex['input_preview']}...")
                print(f"    Order (A,B) winner: {ex['winner_when_AB']}")
                print(f"    Order (B,A) winner: {ex['winner_when_BA']}")
                if ex.get("rationale_AB"):
                    print(f"    Rationale (A,B): {ex['rationale_AB']}")
                if ex.get("rationale_BA"):
                    print(f"    Rationale (B,A): {ex['rationale_BA']}")
            print()
        else:
            print("\n  No flips detected — judge appears position-invariant.\n")

        print(border + "\n")

    # ------------------------------------------------------------------
    # Helper: load pair cases from JSON
    # ------------------------------------------------------------------

    @staticmethod
    def load_pair_cases(path: str | Path) -> list[dict[str, Any]]:
        """Load pairwise test cases from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
