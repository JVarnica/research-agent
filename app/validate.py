"""Validate and retry structured output.

Free generation then pydantic parse to validate if not gives ValidationError,
which is fed back on retry. Guided decoding kept as last resort.

As gen is unconstrained, model no longer recieves JSON schema via response_format. This module
instead injects schema.model_json_schema() into prompt so schema descriptions still reaches model
"""

from __future__ import annotations
import json
import logging
import re
from typing import Type, TypeVar
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

# Qwen3 reasoning block. If vLLM's reasoning parser is enabled the content
# arrives clean and this is a no-op; the regex is defensive either way.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# ```json ... ``` or ``` ... ```
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_SCHEMA_SUFFIX = (
    "\n\---\n"
    "Respond with a SINGLE JSON object that validates against this JSON Schema."
    "Output ONLY the JSON - no prose before or after, no markdown fences.\n"
    "JSON Schema:\n{schema}"
)
_RETRY_TEMPLATE = (
    "Your previous output failed validation with these errors:\n\n"
    "{errors}\n\n"
    "Return the FULL corrected JSON object. Fix ONLY what the errors point at; "
    "keep everything that was already valid. Output only JSON."
)

def _format_errors(e: ValidationError, max_errors: int = 8) -> str:
    """Compact, whole-error formatting. Never string-slice pydantic's repr:
    errors are in document order, so cutting either end hides real problems.
    Instead keep the first N complete errors + a count of the rest — the
    model fixes those, and any remainder surfaces on the next attempt."""
    errs = e.errors(include_url=False)
    lines = []
    for err in errs[:max_errors]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        bad_input = repr(err.get("input"))[:80]
        lines.append(f"- at `{loc}`: {err['msg']} (got: {bad_input})")
    if len(errs) > max_errors:
        lines.append(f"- ...and {len(errs) - max_errors} more errors of the same kinds.")
    return "\n".join(lines)

def extract_json(raw: str) -> str:
    """Best-effort extraction of the JSON payload from free-form output. if reasoning parser chill but just incase"""
    text = _THINK_RE.sub("", raw).strip()
    m = _FENCE_RE.search(text)

    if m:
        text = m.group(1).strip()
        # model might prefix prose.
    if text and text[0] not in "{[":
        start = min(
            (i for i in (text.find("{"), text.find("[")) if i != -1),
            default=-1,
        )
        if start != -1:
            end = max(text.rfind("}"), text.rfind("]"))
            if end > start:
                text = text[start : end + 1]
    return text

def _with_schema(messages: list[dict], schema: Type[T]) -> list[dict]:
    """Append JSON schema to user message (copy, don't mutate)."""
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    msgs = [dict(m) for m in messages]
    if msgs and msgs[-1].get("role") == "user":
        msgs[-1]["content"] = msgs[-1]["content"] + _SCHEMA_SUFFIX.format(schema=schema_json)
    else:
        msgs.append({"role": "user", "content": _SCHEMA_SUFFIX.format(schema=schema_json)})
    return msgs

async def ainvoke_validated(
          llm: BaseChatModel,
          schema: Type[T],
          messages: list[dict],
          *,
          max_attempts: int = 3,
          guided_fallback: BaseChatModel | None = None,
          stats: dict | None = None,
) -> T:
    """Free generation + pydantic validation + validation error feedback retry.
     schema what we validate against
     max_attempts means max 3 retries before guided_fallback which is constrained 
     decoding.
    """

    convo = _with_schema(messages,schema) #appends schema
    last_err: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        resp = await llm.ainvoke(convo)
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        candidate = extract_json(raw)

        try:
            result = schema.model_validate_json(candidate)
            if stats is not None:
                stats.update(attempts=attempt, fallback_used=False)
            if attempt > 1:
                logger.info(
                    "validated %s on attempt %d/%d", schema.__name__, attempt, max_attempts
                )
            return result
        except ValidationError as e:
            last_err = e
            #Pydantic errors on big lists can be huge so truncate need just enough for model to locate issue.
            # Truncate: pydantic errors on big lists can be enormous, and we
            # only need enough for the model to locate the problem.
            err_text = _format_errors(e)
            logger.warning(
                "validation failed for %s (attempt %d/%d): %s",
                schema.__name__, attempt, max_attempts, err_text.splitlines()[0],
            )
            if attempt < max_attempts:
                convo = convo + [
                     {"role": "assistant", "content": _THINK_RE.sub("",raw)},
                     {"role": "user", "content": _RETRY_TEMPLATE.format(errors=err_text)},
                ]
    
    if stats is not None:
        stats.update(attempts=max_attempts, fallback_used=guided_fallback is not None)


    if guided_fallback is not None:
        logger.warning(
            "%s: %d free attempts failed — falling back to guided decoding",
            schema.__name__, max_attempts,
        )
        return await guided_fallback.ainvoke(messages)
    
    raise last_err # type: ignore[misc]

            
