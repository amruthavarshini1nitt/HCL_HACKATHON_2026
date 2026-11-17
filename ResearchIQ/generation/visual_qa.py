"""Step B7 (visual): ask Qwen2.5-VL about the original figure image. Only used for visual questions."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from config import settings
from generation.llm_client import ChatClient, LenientModel, LLMUnavailable, get_vlm
from generation.prompt_templates import VISUAL_QA_SYSTEM
from ingestion.visual_analyzer import image_data_url, image_hash
from retrieval.types import EvidenceRecord

logger = logging.getLogger(__name__)


class VisualAnswer(LenientModel):
    answer: str
    observations: list[str] = Field(default_factory=list)
    uncertain: bool = False
    uncertainty_notes: str | None = None


def _cache_file(img_hash: str, question: str) -> Path:
    d = settings.cache_dir / "vqa"
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{img_hash}|{question}|{settings.vlm_model}".encode()).hexdigest()[:40]
    return d / f"{key}.json"


def ask_about_figure(question: str, rec: EvidenceRecord, client: ChatClient) -> VisualAnswer | None:
    """One VLM call for one figure (cached by image hash + question). None if unavailable."""
    path = Path(rec.asset_path or "")
    if not path.is_file():
        return None
    cf = _cache_file(image_hash(path), question)
    if cf.exists():
        try:
            return VisualAnswer.model_validate_json(cf.read_text(encoding="utf-8"))
        except ValueError:
            cf.unlink(missing_ok=True)
    ctx = (rec.parent_content or "")[:1500]
    messages = [
        {"role": "system", "content": VISUAL_QA_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": f"<question>{question}</question>\n<caption>{(rec.caption or 'none')[:600]}"
                                     f"</caption>\n<context>{ctx or 'none'}</context>"},
            {"type": "image_url", "image_url": {"url": image_data_url(path)}}]}]
    try:
        ans = client.complete_model(VisualAnswer, messages, max_tokens=700)
    except LLMUnavailable as e:
        logger.warning("visual QA unavailable: %s", e)
        return None
    cf.write_text(ans.model_dump_json(indent=1), encoding="utf-8")
    return ans


def visual_evidence(question: str, evidence: list[EvidenceRecord], client: ChatClient | None = None,
                    max_images: int = 2) -> list[EvidenceRecord]:
    """Create extra evidence records holding the VLM's reading of the top figures.

    The new record keeps the figure's paper, page and label; its id is '<figure_id>:VQA' and it is marked
    as model-derived so the answer and UI can treat it as less certain than author text.
    """
    figs = [e for e in evidence if e.element_type in ("figure", "chart") and e.asset_path][:max_images]
    if not figs:
        return []
    client = client or get_vlm()
    out: list[EvidenceRecord] = []
    for f in figs:
        ans = ask_about_figure(question, f, client)
        if ans is None:
            continue
        text = ans.answer + ("\nObservations: " + "; ".join(ans.observations) if ans.observations else "")
        if ans.uncertain:
            text += f"\n(Uncertain: {ans.uncertainty_notes or 'image unclear'})"
        out.append(f.model_copy(update={
            "source_id": f"{f.source_id}:VQA", "content": text, "parent_content": None,
            "derived_from_vlm": True, "retrieval": {"how": "visual_qa", "figure": f.source_id}}))
    return out
