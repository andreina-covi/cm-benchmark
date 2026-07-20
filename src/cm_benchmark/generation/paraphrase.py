"""Optional LLM paraphrase hook (question text only; answer locked).

Off by default. v0 ships a no-op / identity implementation so the CLI flag
exists without requiring an API key.
"""

from __future__ import annotations

from typing import Optional


def paraphrase_question(
    question: str,
    *,
    locked_answer: str,
    options: dict,
    answer_source: list,
    enabled: bool = False,
    model: Optional[str] = None,
) -> str:
    """
    Rewrite question wording without changing the answer.

    When ``enabled`` is False (default), returns ``question`` unchanged.
    A future iteration may call litellm / a provider SDK here; the model must
    not invent spatial facts or alter ``locked_answer``.
    """
    if not enabled:
        return question
    # Placeholder: identity paraphrase so `--paraphrase` is safe without deps.
    _ = (locked_answer, options, answer_source, model)
    return question
