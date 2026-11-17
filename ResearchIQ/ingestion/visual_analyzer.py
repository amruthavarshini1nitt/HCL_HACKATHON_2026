"""Step A5: describe figures with Qwen2.5-VL (hosted), with an on-disk cache keyed by image hash."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from config import settings
from generation.llm_client import ChatClient, LenientModel, LLMUnavailable, get_vlm
from generation.prompt_templates import VISUAL_DESCRIBE_SYSTEM

logger = logging.getLogger(__name__)
PROMPT_VERSION = "v1"


class VisualDescription(LenientModel):
    visual_type: str = "other"
    summary: str = ""
    chart_type: str | None = None
    x_axis: str | None = None
    y_axis: str | None = None
    units: str | None = None
    legend: list[str] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    comparisons: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    connections: list[str] = Field(default_factory=list)
    flow: str | None = None
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    text_in_image: list[str] = Field(default_factory=list)
    uncertain: bool = False
    uncertainty_notes: str | None = None

    @property
    def is_chart(self) -> bool:
        return self.visual_type in ("chart", "graph") or bool(self.chart_type)


def image_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _cache_path(img_hash: str) -> Path:
    d = settings.cache_dir / "vlm"
    d.mkdir(parents=True, exist_ok=True)
    safe_model = settings.vlm_model.replace("/", "_")
    return d / f"{img_hash[:32]}_{safe_model}_{PROMPT_VERSION}.json"


def build_user_prompt(caption: str | None, context: str | None) -> str:
    return (f"<caption>{(caption or 'none')[:800]}</caption>\n<context>{(context or 'none')[:2000]}</context>\n"
            "Describe this figure as the JSON schema requires.")


def describe_image(path: Path, caption: str | None = None, context: str | None = None,
                   client: ChatClient | None = None) -> tuple[VisualDescription | None, bool]:
    """Return (description or None on failure, from_cache). Never raises on API outage."""
    path = Path(path)
    cp = _cache_path(image_hash(path))
    if cp.exists():
        try:
            return VisualDescription.model_validate_json(cp.read_text(encoding="utf-8")), True
        except ValueError:
            cp.unlink(missing_ok=True)
    client = client or get_vlm()
    messages = [
        {"role": "system", "content": VISUAL_DESCRIBE_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": build_user_prompt(caption, context)},
            {"type": "image_url", "image_url": {"url": image_data_url(path)}}]},
    ]
    try:
        desc = client.complete_model(VisualDescription, messages, max_tokens=1200)
    except LLMUnavailable as e:
        logger.warning("VLM unavailable for %s: %s", path.name, e)
        return None, False
    cp.write_text(desc.model_dump_json(indent=1), encoding="utf-8")
    return desc, False


def description_to_text(desc: VisualDescription, label: str | None, caption: str | None) -> str:
    """Searchable text for embedding/BM25: caption plus the structured visual description."""
    parts = [f"{label or 'Figure'}: {caption}" if caption else (label or "Figure"),
             f"Type: {desc.visual_type}" + (f" ({desc.chart_type})" if desc.chart_type else ""),
             desc.summary]
    for name, vals in (("Components", desc.components), ("Connections", desc.connections),
                       ("Legend", desc.legend), ("Trends", desc.trends),
                       ("Comparisons", desc.comparisons), ("Legible values", desc.values),
                       ("Inputs", desc.inputs), ("Outputs", desc.outputs)):
        if vals:
            parts.append(f"{name}: " + "; ".join(vals))
    if desc.x_axis or desc.y_axis:
        parts.append(f"Axes: x={desc.x_axis or '?'}, y={desc.y_axis or '?'}, units={desc.units or 'n/a'}")
    if desc.flow:
        parts.append(f"Flow: {desc.flow}")
    if desc.uncertain:
        parts.append(f"Note (uncertain interpretation): {desc.uncertainty_notes or 'unclear image'}")
    return "\n".join(p for p in parts if p)


def save_description(desc: VisualDescription, element_id: str, paper_id: str) -> Path:
    out = settings.figures_dir / paper_id / f"{element_id}.json"
    out.write_text(json.dumps(desc.model_dump(), indent=1, ensure_ascii=False), encoding="utf-8")
    return out
