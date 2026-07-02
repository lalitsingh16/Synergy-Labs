"""
judge.py — LLM-as-Judge evaluation module.

Loads a rubric definition, constructs a grounded evaluation prompt for each
test case, calls a configurable LLM provider (Anthropic or OpenAI) as the
judge, and returns structured scores with per-criterion rationale.

Key features:
- ``parse_verdict()``: Robust JSON extraction with regex repair and auto-retry.
- JSONL call logging: Every judge call (prompt, response, verdict, token usage)
  is appended to ``logs/judge_log.jsonl``.
- ``run_suite()``: Evaluates all test cases in a test suite file.
- ``aggregate_report()``: Computes pass rate, per-criterion means, and writes
  ``reports/suite_report.json``.

Usage:
    from src.judge import Judge

    judge = Judge(
        rubric_path="data/rubric.json",
        provider="anthropic",                       # or "openai"
        model="claude-3-5-sonnet-20241022",
        api_key="sk-ant-...",
    )

    verdicts = judge.run_suite("data/test_suite.json")
    report   = judge.aggregate_report(verdicts)
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# pyrefly: ignore [missing-import]
import anthropic
# pyrefly: ignore [missing-import]
import openai
# pyrefly: ignore [missing-import]
import groq
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TokenUsage:
    """Token counts from a single judge API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CriterionResult:
    """Score and rationale for a single evaluation criterion."""

    name: str
    score: int
    rationale: str
    weight: float = 0.0


@dataclass
class JudgeResult:
    """Full evaluation result for one test case."""

    test_case_id: str
    criterion_scores: dict[str, int] = field(default_factory=dict)
    rationale: dict[str, str] = field(default_factory=dict)
    overall_score: float = 0.0
    criteria_details: list[CriterionResult] = field(default_factory=list)
    raw_response: str = ""
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    error: str | None = None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert LLM evaluator. Your task is to assess the quality of a \
model-generated response against a provided rubric.

## Rules — follow these exactly:

1. **Score each criterion independently** using the 1–5 integer scale defined \
in the rubric below. Only score the criteria explicitly listed in the \
evaluation request — ignore all others.
2. **Ground every score** with a single concise sentence that cites *specific \
evidence* from the model output. Do NOT give generic praise or criticism; \
quote or reference concrete words, phrases, claims, or structural elements.
3. **Be strict and calibrated.** A score of 5 means genuinely excellent — \
reserve it for outputs with no identifiable weakness on that criterion. \
A score of 1 means severely deficient. Use the full range.
4. **Compute `overall_score`** as the weighted average of all scored criteria, \
rounded to two decimal places. Use the weights provided in the rubric.
5. **Return ONLY valid JSON** — no markdown fences, no commentary outside the \
JSON object.

## Required JSON schema:

```
{
  "criterion_scores": { "<criterion_name>": <int 1-5>, ... },
  "rationale":        { "<criterion_name>": "<one-sentence evidence-based justification>", ... },
  "overall_score":    <float, weighted average rounded to 2 decimals>
}
```
"""


def _format_rubric_section(rubric: dict[str, Any], criteria: list[str]) -> str:
    """Format only the requested criteria from the rubric into readable text."""
    lines: list[str] = ["## Scoring Rubric\n"]

    all_criteria = rubric.get("criteria", {})
    for criterion_name in criteria:
        criterion = all_criteria.get(criterion_name)
        if criterion is None:
            logger.warning(
                "Criterion '%s' not found in rubric — skipping.", criterion_name
            )
            continue

        lines.append(f"### {criterion_name}  (weight: {criterion['weight']})")
        lines.append(f"{criterion['description']}\n")
        for level, description in sorted(criterion["scale"].items()):
            lines.append(f"  {level}: {description}")
        lines.append("")  # blank line between criteria

    return "\n".join(lines)


def _format_evaluation_request(
    *,
    user_input: str,
    system_prompt: str,
    model_output: str,
    expected_output: str | None,
    rubric_section: str,
) -> str:
    """Build the user-turn message containing the test case and rubric."""
    output_display = model_output.strip() if model_output and model_output.strip() else "[EMPTY RESPONSE / NO OUTPUT GENERATED]"
    parts: list[str] = [
        "Evaluate the following model output.\n",
        "---",
        f"**System prompt given to the model:**\n{system_prompt}\n",
        f"**User input:**\n{user_input}\n",
        f"**Model output to evaluate:**\n{output_display}\n",
    ]

    if expected_output:
        parts.append(
            f"**Reference / expected output (use as a guide, not a rigid template):**"
            f"\n{expected_output}\n"
        )

    parts.append("---\n")
    parts.append(rubric_section)
    parts.append(
        "\nScore each listed criterion and return ONLY the JSON object described above."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JSON parsing / verdict extraction
# ---------------------------------------------------------------------------
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")

# Prompt appended on the retry call when the first response isn't valid JSON.
_JSON_RETRY_NUDGE = (
    "Your previous response was not valid JSON. "
    "Return ONLY a single raw JSON object (no markdown fences, no commentary). "
    "Use the exact schema specified in the original instructions."
)


def parse_verdict(raw_text: str) -> dict[str, Any]:
    """
    Extract a structured verdict dict from raw judge output.

    Applies three increasingly aggressive strategies:
      1. ``json.loads()`` on the raw text.
      2. Strip markdown fences and retry.
      3. Locate the outermost ``{ … }`` via brace-depth scan and retry.

    Parameters
    ----------
    raw_text : str
        The raw text returned by the judge LLM.

    Returns
    -------
    dict
        Parsed JSON verdict.

    Raises
    ------
    ValueError
        If none of the strategies produce valid JSON.
    """
    text = raw_text.strip()

    # --- Strategy 1: direct parse ----------------------------------------
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # --- Strategy 2: strip markdown fences --------------------------------
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # --- Strategy 3: outermost { … } brace extraction ---------------------
    #   Walk from the first '{' and track brace depth to find the matching '}'.
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # --- Fallback: simple rfind as last resort ----------------------------
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract valid JSON from judge response:\n{text[:500]}"
    )


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------

@dataclass
class _LLMResponse:
    """Internal wrapper: raw text + token counts from a single API call."""

    text: str
    usage: TokenUsage


def _call_anthropic(
    *,
    system_prompt: str,
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> _LLMResponse:
    """Call the Anthropic Messages API and return text + token usage."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
        )
        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        usage = TokenUsage(
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
            total_tokens=(
                getattr(response.usage, "input_tokens", 0)
                + getattr(response.usage, "output_tokens", 0)
            ),
        )
        return _LLMResponse(text=text, usage=usage)
    except Exception as exc:
        logger.error(
            "Anthropic API call failed for model '%s'. Reason: %s",
            model,
            exc,
            exc_info=True
        )
        raise RuntimeError(f"Anthropic API call failed: {exc}") from exc


def _call_openai(
    *,
    system_prompt: str,
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> _LLMResponse:
    """Call the OpenAI Chat Completions API and return text + token usage."""
    try:
        client = openai.OpenAI(api_key=api_key)
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=full_messages,
        )
        text = response.choices[0].message.content or ""
        tok = response.usage
        usage = TokenUsage(
            input_tokens=getattr(tok, "prompt_tokens", 0) if tok else 0,
            output_tokens=getattr(tok, "completion_tokens", 0) if tok else 0,
            total_tokens=getattr(tok, "total_tokens", 0) if tok else 0,
        )
        return _LLMResponse(text=text, usage=usage)
    except Exception as exc:
        logger.error(
            "OpenAI API call failed for model '%s'. Reason: %s",
            model,
            exc,
            exc_info=True
        )
        raise RuntimeError(f"OpenAI API call failed: {exc}") from exc


def _call_groq(
    *,
    system_prompt: str,
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> _LLMResponse:
    """Call the Groq Chat Completions API and return text + token usage."""
    try:
        client = groq.Groq(api_key=api_key)
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=full_messages,
        )
        text = response.choices[0].message.content or ""
        tok = response.usage
        usage = TokenUsage(
            input_tokens=getattr(tok, "prompt_tokens", 0) if tok else 0,
            output_tokens=getattr(tok, "completion_tokens", 0) if tok else 0,
            total_tokens=getattr(tok, "total_tokens", 0) if tok else 0,
        )
        return _LLMResponse(text=text, usage=usage)
    except Exception as exc:
        logger.error(
            "Groq API call failed for model '%s'. Reason: %s",
            model,
            exc,
            exc_info=True
        )
        raise RuntimeError(f"Groq API call failed: {exc}") from exc


def _call_google(
    *,
    system_prompt: str,
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> _LLMResponse:
    """Call the Google GenAI API and return text + token usage."""
    try:
        client = genai.Client(api_key=api_key)
        
        # Transform messages: translate role "assistant" to "model" and wrap content in parts
        google_contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            google_contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
            
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
        
        response = client.models.generate_content(
            model=model,
            contents=google_contents,
            config=config,
        )
        
        text = response.text or ""
        tok = response.usage_metadata
        usage = TokenUsage(
            input_tokens=getattr(tok, "prompt_token_count", 0) if tok else 0,
            output_tokens=getattr(tok, "candidates_token_count", 0) if tok else 0,
            total_tokens=getattr(tok, "total_token_count", 0) if tok else 0,
        )
        return _LLMResponse(text=text, usage=usage)
    except Exception as exc:
        logger.error(
            "Google GenAI API call failed for model '%s'. Reason: %s",
            model,
            exc,
            exc_info=True
        )
        raise RuntimeError(f"Google GenAI API call failed: {exc}") from exc


_PROVIDERS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "groq": _call_groq,
    "google": _call_google,
}


# ---------------------------------------------------------------------------
# Judge class
# ---------------------------------------------------------------------------
class Judge:
    """
    LLM-as-Judge evaluator.

    Parameters
    ----------
    rubric_path : str | Path
        Path to rubric.json.
    provider : str
        LLM provider — ``"anthropic"`` or ``"openai"``.
    model : str
        Model identifier (e.g. ``"claude-3-5-sonnet-20241022"``).
    api_key : str
        API key for the chosen provider.
    temperature : float
        Sampling temperature for the judge (default 0.0 for determinism).
    max_tokens : int
        Maximum tokens in the judge response.
    log_dir : str | Path
        Directory for the JSONL call log (default ``"logs"``).
    """

    def __init__(
        self,
        rubric_path: str | Path,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        log_dir: str | Path = "logs",
    ) -> None:
        import os
        from dotenv import load_dotenv
        load_dotenv()

        if model is None:
            model = os.getenv("JUDGE_MODEL")
            if not model:
                model = os.getenv("ANTHROPIC_JUDGE_MODEL") or "claude-3-5-sonnet-20241022"

        if provider is None:
            provider = os.getenv("JUDGE_PROVIDER")
            if not provider:
                # auto-detect based on model name
                if model and ("claude" in model.lower() or "anthropic" in model.lower()):
                    provider = "anthropic"
                elif model and ("gpt" in model.lower() or "o1" in model.lower() or "o3" in model.lower() or "openai" in model.lower()):
                    provider = "openai"
                elif model and ("gemini" in model.lower() or "google" in model.lower()):
                    provider = "google"
                elif model and ("llama" in model.lower() or "mixtral" in model.lower() or "gemma" in model.lower() or "groq" in model.lower()):
                    provider = "groq"
                else:
                    provider = "anthropic"

        self.provider = provider.lower()
        if self.provider not in _PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{provider}'. Choose from: {list(_PROVIDERS)}"
            )

        self.model = model

        if api_key is None:
            if self.provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "google":
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            elif self.provider == "groq":
                api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            expected_keys = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "google": "GEMINI_API_KEY or GOOGLE_API_KEY",
                "groq": "GROQ_API_KEY",
            }
            raise ValueError(
                f"API key not found for provider '{self.provider}'. "
                f"Please configure {expected_keys.get(self.provider, 'relevant API key')} in your .env file."
            )

        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load and cache rubric.
        rubric_path = Path(rubric_path)
        if not rubric_path.exists():
            raise FileNotFoundError(f"Rubric not found at {rubric_path}")
        with open(rubric_path, "r", encoding="utf-8") as f:
            self.rubric: dict[str, Any] = json.load(f)

        # Ensure log directory exists.
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / "judge_log.jsonl"

        logger.info(
            "Judge initialised — provider=%s, model=%s, rubric=%s, log=%s",
            self.provider,
            self.model,
            rubric_path.name,
            self._log_path,
        )

    # ------------------------------------------------------------------
    # Internal: LLM call dispatcher
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = _JUDGE_SYSTEM_PROMPT,
    ) -> _LLMResponse:
        """Dispatch a call to the configured judge LLM provider."""
        call_fn = _PROVIDERS[self.provider]
        return call_fn(
            system_prompt=system_prompt,
            messages=messages,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    # ------------------------------------------------------------------
    # JSONL call logger
    # ------------------------------------------------------------------

    def _log_call(
        self,
        *,
        test_case_id: str,
        judge_prompt: str,
        raw_response: str,
        parsed_verdict: dict[str, Any] | None,
        token_usage: TokenUsage,
        attempt: int,
        error: str | None = None,
    ) -> None:
        """
        Append a single log entry to ``logs/judge_log.jsonl``.

        Each line is a self-contained JSON object with a UTC timestamp.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_case_id": test_case_id,
            "provider": self.provider,
            "model": self.model,
            "attempt": attempt,
            "judge_prompt": judge_prompt,
            "raw_response": raw_response,
            "parsed_verdict": parsed_verdict,
            "token_usage": {
                "input_tokens": token_usage.input_tokens,
                "output_tokens": token_usage.output_tokens,
                "total_tokens": token_usage.total_tokens,
            },
            "error": error,
        }
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write judge log: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, test_case: dict[str, Any]) -> JudgeResult:
        """
        Evaluate a single test case against the rubric.

        Flow:
          1. Build the rubric section & evaluation prompt.
          2. Call the judge LLM.
          3. Attempt to parse the response with ``parse_verdict()``.
          4. On parse failure, retry the API call **once** with a nudge
             message asking for valid JSON.
          5. Log every call (prompt, raw response, verdict, token usage)
             to ``logs/judge_log.jsonl``.

        Parameters
        ----------
        test_case : dict
            Must contain keys: ``input``, ``system_prompt``,
            ``model_output``, ``criteria``.
            Optional: ``id``, ``expected_output``.

        Returns
        -------
        JudgeResult
            Structured evaluation result with scores, rationales,
            weighted overall score, and the raw LLM response.
        """
        test_case_id = test_case.get("id", "unknown")
        criteria: list[str] = test_case["criteria"]

        # 1. Build rubric section for only the requested criteria.
        rubric_section = _format_rubric_section(self.rubric, criteria)

        # 2. Build the full evaluation prompt (user turn).
        user_message = _format_evaluation_request(
            user_input=test_case["input"],
            system_prompt=test_case["system_prompt"],
            model_output=test_case["model_output"],
            expected_output=test_case.get("expected_output"),
            rubric_section=rubric_section,
        )

        # 3. Call the judge LLM (attempt 1).
        messages: list[dict[str, str]] = [
            {"role": "user", "content": user_message},
        ]
        cumulative_usage = TokenUsage()

        try:
            llm_resp = self._call_llm(messages)
        except Exception as exc:
            logger.error("Judge API call failed for %s: %s", test_case_id, exc)
            self._log_call(
                test_case_id=test_case_id,
                judge_prompt=user_message,
                raw_response="",
                parsed_verdict=None,
                token_usage=TokenUsage(),
                attempt=1,
                error=str(exc),
            )
            return JudgeResult(
                test_case_id=test_case_id,
                error=f"API call failed: {exc}",
            )

        raw_response = llm_resp.text
        cumulative_usage = llm_resp.usage

        # 4. Parse verdict — with one auto-retry on failure.
        parsed: dict[str, Any] | None = None
        parse_error: str | None = None

        try:
            parsed = parse_verdict(raw_response)
        except ValueError as exc:
            parse_error = str(exc)
            logger.warning(
                "parse_verdict failed for %s (attempt 1), retrying: %s",
                test_case_id,
                exc,
            )

        # Log attempt 1.
        self._log_call(
            test_case_id=test_case_id,
            judge_prompt=user_message,
            raw_response=raw_response,
            parsed_verdict=parsed,
            token_usage=llm_resp.usage,
            attempt=1,
            error=parse_error,
        )

        # --- Retry once if parse failed -----------------------------------
        if parsed is None:
            retry_messages = messages + [
                {"role": "assistant", "content": raw_response},
                {"role": "user", "content": _JSON_RETRY_NUDGE},
            ]
            llm_resp_2: _LLMResponse | None = None
            try:
                llm_resp_2 = self._call_llm(retry_messages)
                raw_response = llm_resp_2.text
                cumulative_usage = TokenUsage(
                    input_tokens=(
                        cumulative_usage.input_tokens
                        + llm_resp_2.usage.input_tokens
                    ),
                    output_tokens=(
                        cumulative_usage.output_tokens
                        + llm_resp_2.usage.output_tokens
                    ),
                    total_tokens=(
                        cumulative_usage.total_tokens
                        + llm_resp_2.usage.total_tokens
                    ),
                )
                parsed = parse_verdict(raw_response)
                parse_error = None
            except (ValueError, Exception) as exc:
                parse_error = f"Retry also failed: {exc}"
                logger.error(
                    "parse_verdict retry failed for %s: %s",
                    test_case_id,
                    exc,
                )

            # Log attempt 2.
            self._log_call(
                test_case_id=test_case_id,
                judge_prompt=_JSON_RETRY_NUDGE,
                raw_response=raw_response,
                parsed_verdict=parsed,
                token_usage=(
                    llm_resp_2.usage if llm_resp_2 is not None
                    else TokenUsage()
                ),
                attempt=2,
                error=parse_error,
            )

        # If still no valid parse, return an error result.
        if parsed is None:
            # check if the raw response looks like a model refusal
            is_refusal = any(
                kw in raw_response.lower()
                for kw in ["i cannot", "i apologize", "sorry, but", "unethical", "unable to process", "as an ai"]
            )
            if is_refusal:
                error_msg = f"Judge model refused to evaluate this case: {raw_response[:200]}"
            else:
                error_msg = f"JSON parse failed after retry: {parse_error}"

            return JudgeResult(
                test_case_id=test_case_id,
                raw_response=raw_response,
                token_usage=cumulative_usage,
                error=error_msg,
            )

        # 5. Assemble the result.
        criterion_scores = parsed.get("criterion_scores", {})
        rationale = parsed.get("rationale", {})

        # Build detailed per-criterion results and compute weighted average.
        criteria_details: list[CriterionResult] = []
        weighted_sum = 0.0
        total_weight = 0.0

        all_criteria_meta = self.rubric.get("criteria", {})
        for name in criteria:
            score = criterion_scores.get(name)
            if score is None:
                logger.warning(
                    "Judge did not return a score for '%s' in %s.",
                    name,
                    test_case_id,
                )
                continue

            weight = all_criteria_meta.get(name, {}).get("weight", 0.0)
            criteria_details.append(
                CriterionResult(
                    name=name,
                    score=int(score),
                    rationale=rationale.get(name, ""),
                    weight=weight,
                )
            )
            weighted_sum += int(score) * weight
            total_weight += weight

        # Normalise if not all 5 criteria were scored (partial weight).
        overall_score = round(weighted_sum / total_weight, 2) if total_weight else 0.0

        return JudgeResult(
            test_case_id=test_case_id,
            criterion_scores={c.name: c.score for c in criteria_details},
            rationale={c.name: c.rationale for c in criteria_details},
            overall_score=overall_score,
            criteria_details=criteria_details,
            raw_response=raw_response,
            token_usage=cumulative_usage,
        )

    def evaluate_batch(
        self, test_cases: list[dict[str, Any]]
    ) -> list[JudgeResult]:
        """
        Evaluate multiple test cases sequentially.

        Parameters
        ----------
        test_cases : list[dict]
            List of test case dicts (same schema as ``evaluate``).

        Returns
        -------
        list[JudgeResult]
        """
        results: list[JudgeResult] = []
        for i, tc in enumerate(test_cases, 1):
            tc_id = tc.get("id", f"case_{i}")
            logger.info("Evaluating %s (%d/%d)...", tc_id, i, len(test_cases))
            results.append(self.evaluate(tc))
        return results

    # ------------------------------------------------------------------
    # Suite-level evaluation
    # ------------------------------------------------------------------

    def run_suite(
        self,
        test_suite_path: str | Path,
    ) -> list[JudgeResult]:
        """
        Load a test suite and evaluate every case.

        Parameters
        ----------
        test_suite_path : str | Path
            Path to a JSON file containing a list of test case dicts.

        Returns
        -------
        list[JudgeResult]
            One result per test case, in order.
        """
        test_cases = self.load_test_suite(test_suite_path)
        logger.info(
            "Running suite: %d test cases from %s",
            len(test_cases),
            test_suite_path,
        )

        results: list[JudgeResult] = []
        for i, tc in enumerate(test_cases, 1):
            tc_id = tc.get("id", f"case_{i}")
            logger.info("Evaluating %s (%d/%d)...", tc_id, i, len(test_cases))
            results.append(self.evaluate(tc))

        passed = sum(1 for r in results if self.did_pass(r))
        logger.info(
            "Suite complete: %d/%d passed (threshold %.2f)",
            passed,
            len(results),
            self.get_pass_threshold(),
        )
        return results

    # ------------------------------------------------------------------
    # Aggregate reporting
    # ------------------------------------------------------------------

    def aggregate_report(
        self,
        results: list[JudgeResult],
        report_dir: str | Path = "reports",
    ) -> dict[str, Any]:
        """
        Compute suite-level statistics and write ``suite_report.json``.

        Computes:
          - **pass_rate**: fraction of test cases with
            ``overall_score >= threshold``.
          - **mean_overall_score**: arithmetic mean of all overall scores.
          - **mean_score_per_criterion**: mean of each criterion across
            all test cases that included it.
          - **per_case_summary**: condensed per-case scores and pass/fail.

        Parameters
        ----------
        results : list[JudgeResult]
            Output of ``run_suite()`` or ``evaluate()`` calls.
        report_dir : str | Path
            Directory to write ``suite_report.json`` into.

        Returns
        -------
        dict
            The full report as a dict (also written to disk).
        """
        threshold = self.get_pass_threshold()
        total = len(results)

        if total == 0:
            logger.warning("aggregate_report called with 0 results.")
            return {"error": "No results to aggregate."}

        # --- Pass rate ----------------------------------------------------
        num_passed = sum(1 for r in results if self.did_pass(r))
        pass_rate = round(num_passed / total, 4)

        # --- Mean overall score -------------------------------------------
        overall_scores = [r.overall_score for r in results if r.error is None]
        mean_overall = (
            round(sum(overall_scores) / len(overall_scores), 4)
            if overall_scores
            else 0.0
        )

        # --- Mean per criterion -------------------------------------------
        criterion_accum: dict[str, list[int]] = defaultdict(list)
        for r in results:
            for name, score in r.criterion_scores.items():
                criterion_accum[name].append(score)

        mean_per_criterion = {
            name: round(sum(scores) / len(scores), 4)
            for name, scores in sorted(criterion_accum.items())
        }

        # --- Per-case summary ---------------------------------------------
        per_case: list[dict[str, Any]] = []
        for r in results:
            per_case.append(
                {
                    "test_case_id": r.test_case_id,
                    "overall_score": r.overall_score,
                    "passed": self.did_pass(r),
                    "criterion_scores": r.criterion_scores,
                    "error": r.error,
                }
            )

        # --- Token usage totals -------------------------------------------
        total_input_tokens = sum(r.token_usage.input_tokens for r in results)
        total_output_tokens = sum(r.token_usage.output_tokens for r in results)

        # --- Assemble report ----------------------------------------------
        report: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "judge_provider": self.provider,
            "judge_model": self.model,
            "pass_threshold": threshold,
            "total_cases": total,
            "passed": num_passed,
            "failed": total - num_passed,
            "pass_rate": pass_rate,
            "mean_overall_score": mean_overall,
            "mean_score_per_criterion": mean_per_criterion,
            "total_token_usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            },
            "per_case_summary": per_case,
        }

        # --- Write to disk ------------------------------------------------
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        out_file = report_path / "suite_report.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Suite report written to %s", out_file)
        return report

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_pass_threshold(self) -> float:
        """Return the pass/fail threshold from the rubric's scoring_notes."""
        return self.rubric.get("scoring_notes", {}).get("pass_threshold", 3.5)

    def did_pass(self, result: JudgeResult) -> bool:
        """Check whether a JudgeResult meets the rubric's pass threshold."""
        return result.overall_score >= self.get_pass_threshold()

    @staticmethod
    def load_test_suite(path: str | Path) -> list[dict[str, Any]]:
        """Load test cases from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def result_to_dict(self, result: JudgeResult) -> dict[str, Any]:
        """Serialise a JudgeResult to a plain dict for JSON export."""
        return {
            "test_case_id": result.test_case_id,
            "overall_score": result.overall_score,
            "passed": self.did_pass(result),
            "criterion_scores": result.criterion_scores,
            "rationale": result.rationale,
            "criteria_details": [
                {
                    "name": c.name,
                    "score": c.score,
                    "weight": c.weight,
                    "rationale": c.rationale,
                }
                for c in result.criteria_details
            ],
            "token_usage": {
                "input_tokens": result.token_usage.input_tokens,
                "output_tokens": result.token_usage.output_tokens,
                "total_tokens": result.token_usage.total_tokens,
            },
            "error": result.error,
        }
