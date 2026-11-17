"""Thin OpenAI-compatible chat client used for both the LLM and the VLM (provider configurable)."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, TypeVar, get_origin

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError, model_validator

from config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LenientModel(BaseModel):
    """Base for LLM-output schemas: models often send null for empty lists; treat it as []."""

    @model_validator(mode="before")
    @classmethod
    def _nulls_to_empty_lists(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for name, field in cls.model_fields.items():
                if get_origin(field.annotation) is not list or name not in data:
                    continue
                val = data[name]
                if val is None or val == "":
                    data[name] = []
                elif isinstance(val, (str, int, float)):  # single scalar where a list is expected
                    data[name] = [val]
        return data


class LLMUnavailable(Exception):
    """The hosted model could not be reached or returned unusable output. Message is user-safe."""


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model text, tolerating code fences and leading prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj


class ChatClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: int, name: str = "llm") -> None:
        self.model, self.name, self.timeout = model, name, timeout
        self.available = bool(api_key)
        self._client = OpenAI(api_key=api_key or "missing", base_url=base_url, timeout=timeout,
                              max_retries=0) if api_key else None

    def complete(self, messages: list[dict[str, Any]], *, temperature: float = 0.0,
                 max_tokens: int = 1500, json_mode: bool = False, retries: int = 2) -> str:
        if not self._client:
            raise LLMUnavailable(f"The {self.name} is not configured (missing API key).")
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages,
                                  "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content if resp.choices else None
                if not content:
                    raise LLMUnavailable(f"The {self.name} returned an empty response.")
                return content
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last = e
            except APIStatusError as e:
                last = e
                if e.status_code < 500 and e.status_code != 429:
                    if json_mode and e.status_code == 400:  # provider may not support json_object
                        kwargs.pop("response_format", None)
                        json_mode = False
                        continue
                    break
            time.sleep(min(2 ** attempt, 8))
        logger.error("%s call failed: %s", self.name, type(last).__name__)
        raise LLMUnavailable(f"The {self.name} service is currently unavailable.") from None

    def complete_model(self, schema: type[T], messages: list[dict[str, Any]], *, max_tokens: int = 1500,
                       temperature: float = 0.0, parse_retries: int = 1) -> T:
        """Call the model and validate its JSON against a Pydantic schema (one repair retry)."""
        msgs = list(messages)
        for attempt in range(parse_retries + 1):
            text = self.complete(msgs, json_mode=True, max_tokens=max_tokens, temperature=temperature)
            try:
                return schema.model_validate(extract_json(text))
            except (ValueError, ValidationError) as e:
                logger.warning("%s returned invalid JSON (attempt %d): %s", self.name, attempt + 1,
                               type(e).__name__)
                msgs = msgs + [{"role": "assistant", "content": text[:2000]},
                               {"role": "user", "content": "Your reply was not valid JSON for the required "
                                "schema. Reply again with ONLY the JSON object."}]
        raise LLMUnavailable(f"The {self.name} returned malformed output.")


def get_llm() -> ChatClient:
    return ChatClient(settings.llm_api_key, settings.llm_base_url, settings.llm_model,
                      settings.llm_timeout_s, "language model")


def get_vlm() -> ChatClient:
    return ChatClient(settings.vlm_api_key, settings.vlm_base_url, settings.vlm_model,
                      settings.vlm_timeout_s, "vision model")
