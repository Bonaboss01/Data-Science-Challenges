"""
genai_experiments.py

A lightweight GenAI experimentation runner:
- Runs prompt experiments (A/B prompts or prompt variants)
- Logs configs + outputs + optional human scores to JSONL
- Supports offline "mock" mode so the script runs without an API key

This is a production-friendly pattern used in real GenAI teams:
prompt versioning + experiment logs + repeatability.

Run:
    python genai_experiments.py

Optional env vars:
    GENAI_MODE=mock                 # default: mock (no API calls)
    GENAI_MODEL=gpt-4.1-mini        # any string, for logging consistency
    GENAI_LOG_PATH=logs/genai_runs.jsonl
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
import json
import os


# -------------------------
# Config
# -------------------------

DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_PATH = os.getenv("GENAI_LOG_PATH", os.path.join(DEFAULT_LOG_DIR, "genai_runs.jsonl"))
GENAI_MODE = os.getenv("GENAI_MODE", "mock").lower()
GENAI_MODEL = os.getenv("GENAI_MODEL", "gpt-4.1-mini")


@dataclass
class PromptVariant:
    """One prompt variant to test."""
    name: str
    system: str
    user_template: str


@dataclass
class RunConfig:
    """LLM call / experiment parameters."""
    model: str = GENAI_MODEL
    temperature: float = 0.2
    max_tokens: int = 300
    top_p: float = 1.0
    seed: Optional[int] = 42


def _ensure_log_path(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _mock_llm(system: str, user: str, cfg: RunConfig) -> str:
    """
    Offline stand-in output so this file runs without external dependencies.
    """
    # Simple deterministic "response" for experimentation scaffolding
    return (
        f"[MOCK OUTPUT | model={cfg.model} temp={cfg.temperature}] "
        f"Summary: {user[:160]}..."
    )


def call_llm(system: str, user: str, cfg: RunConfig) -> str:
    """
    LLM call wrapper. By default uses mock mode.
    Replace the implementation with your preferred SDK later.
    """
    if GENAI_MODE == "mock":
        return _mock_llm(system, user, cfg)

    # ---- REAL MODE PLACEHOLDER (safe stub) ----
    # To keep this repo clean and runnable, we do not hardcode any API calls here.
    # You can implement:
    # - OpenAI Responses API
    # - Azure OpenAI
    # - Bedrock
    # - Vertex AI
    #
    # For now, raise a helpful error:
    raise RuntimeError(
        "GENAI_MODE is not 'mock' but no provider is configured. "
        "Set GENAI_MODE=mock or implement call_llm() for your provider."
    )


def render_prompt(template: str, variables: Dict[str, Any]) -> str:
    """
    Simple template rendering using Python format.
    Example: 'Explain {metric} for {audience}'.
    """
    try:
        return template.format(**variables)
    except KeyError as e:
        missing = str(e).strip("'")
        raise KeyError(f"Missing template variable: {missing}") from e


def log_run(record: Dict[str, Any], log_path: str = DEFAULT_LOG_PATH) -> None:
    _ensure_log_path(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_experiment(
    variants: List[PromptVariant],
    dataset: List[Dict[str, Any]],
    cfg: RunConfig,
    notes: str = "",
    log_path: str = DEFAULT_LOG_PATH,
) -> List[Dict[str, Any]]:
    """
    Run each prompt variant against each dataset row.
    Returns all run records (also written to JSONL).
    """
    results: List[Dict[str, Any]] = []

    for row_idx, row in enumerate(dataset):
        for v in variants:
            user_prompt = render_prompt(v.user_template, row)

            output = call_llm(v.system, user_prompt, cfg)

            record = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "mode": GENAI_MODE,
                "model": cfg.model,
                "params": asdict(cfg),
                "variant_name": v.name,
                "system_hash": _hash(v.system),
                "prompt_hash": _hash(user_prompt),
                "row_idx": row_idx,
                "input": row,                       # for reproducibility
                "user_prompt": user_prompt,         # store full prompt (ok for small data)
                "output": output,
                "human_score": None,                # fill later (0-5)
                "human_notes": "",
                "notes": notes,
            }

            log_run(record, log_path=log_path)
            results.append(record)

    return results


def score_run(log_path: str, prompt_hash: str, score: int, notes: str = "") -> None:
    """
    Minimal human-in-the-loop scoring:
    Append a 'score' event for a given prompt_hash.
    (Keeps logs append-only; easy for audit trails.)
    """
    if not (0 <= score <= 5):
        raise ValueError("Score must be between 0 and 5.")

    record = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "event": "human_score",
        "prompt_hash": prompt_hash,
        "score": score,
        "notes": notes,
    }
    log_run(record, log_path=log_path)


if __name__ == "__main__":
    # -------------------------
    # Example: experiment dataset
    # -------------------------
    dataset = [
        {"metric": "RMSE", "audience": "a non-technical stakeholder", "context": "weekly sales forecasting"},
        {"metric": "MAPE", "audience": "a finance manager", "context": "monthly revenue forecast"},
    ]

    # -------------------------
    # Example: prompt variants (A/B)
    # -------------------------
    variants = [
        PromptVariant(
            name="A_simple",
            system="You are a helpful data scientist. Keep it simple and clear.",
            user_template="Explain {metric} in the context of {context} to {audience}. Give a 2-sentence explanation.",
        ),
        PromptVariant(
            name="B_actionable",
            system="You are a senior analytics lead. Be concise and action-oriented.",
            user_template=(
                "Explain {metric} for {context} to {audience}. "
                "Use 1 sentence definition + 2 bullet points on how to improve it."
            ),
        ),
    ]

    cfg = RunConfig(model=GENAI_MODEL, temperature=0.2, max_tokens=250)

    runs = run_experiment(
        variants=variants,
        dataset=dataset,
        cfg=cfg,
        notes="Initial GenAI prompt A/B for metric explanations",
        log_path=DEFAULT_LOG_PATH,
    )

    print(f"Completed {len(runs)} runs. Logged to: {DEFAULT_LOG_PATH}")
    print("Tip: open the JSONL file and manually add human_score via score_run().")
