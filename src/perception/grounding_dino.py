"""
Grounding DINO Perception Module
=================================
Open-vocabulary object detection powered by Grounding DINO
(Liu et al., 2023 – https://arxiv.org/abs/2303.05499).

Given an image and a free-form text prompt (e.g. "red car . person . bicycle"),
the model predicts bounding boxes with confidence scores for every mentioned
category – no predefined label set required.

Architecture summary
--------------------
Grounding DINO is a detector that fuses a text encoder (BERT) with an image
encoder (Swin Transformer) via a feature-enhancer and a decoder that attends
jointly to image tokens and text tokens.  The final matching is performed with
a bipartite Hungarian matching loss.

Detection output format (per box)::

    {
        "label": str,          # matched text phrase
        "confidence": float,   # 0–1
        "box": [x1, y1, x2, y2]  # pixel coordinates (absolute)
    }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from PIL import Image as PILImage
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning("torch/PIL not installed – GroundingDINODetector runs in stub mode.")

try:
    from groundingdino.util.inference import (
        load_model,
        load_image,
        predict,
        annotate,
    )
    _GDINO_AVAILABLE = True
except ImportError:
    _GDINO_AVAILABLE = False
    logger.warning(
        "groundingdino not installed – GroundingDINODetector runs in stub mode."
    )


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class DetectionResult:
    """A single detected object."""

    def __init__(
        self,
        label: str,
        confidence: float,
        box: list[float],
    ) -> None:
        if len(box) != 4:
            raise ValueError("box must have exactly 4 elements [x1, y1, x2, y2]")
        self.label = label
        self.confidence = confidence
        self.box = box  # [x1, y1, x2, y2] in pixels

    @property
    def center(self) -> tuple[float, float]:
        """Return (cx, cy) in pixels."""
        return (
            (self.box[0] + self.box[2]) / 2,
            (self.box[1] + self.box[3]) / 2,
        )

    @property
    def area(self) -> float:
        """Return bounding-box area in square pixels."""
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "box": [round(v, 1) for v in self.box],
            "center": [round(v, 1) for v in self.center],
            "area": round(self.area, 1),
        }

    def __repr__(self) -> str:
        return (
            f"DetectionResult(label={self.label!r}, "
            f"confidence={self.confidence:.3f}, box={self.box})"
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class GroundingDINODetector:
    """Open-vocabulary object detector using Grounding DINO.

    Parameters
    ----------
    config_path:
        Path to the GroundingDINO model config (Python file).
    weights_path:
        Path to the pretrained model weights (.pth file).
    box_threshold:
        Minimum confidence score for a bounding box to be kept.
    text_threshold:
        Minimum token-level confidence for a phrase to be associated
        with a box.
    device:
        PyTorch device string ("cpu", "cuda", "cuda:0", …).
    """

    def __init__(
        self,
        config_path: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        weights_path: str = "weights/groundingdino_swint_ogc.pth",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self.config_path = config_path
        self.weights_path = weights_path
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load the model weights into memory.

        This is intentionally separated from ``__init__`` so the object can be
        created cheaply (e.g. in tests) and the heavy load deferred.
        """
        if not _GDINO_AVAILABLE:
            logger.info("[STUB] GroundingDINO model NOT loaded (stub mode).")
            return

        logger.info(
            "Loading Grounding DINO from %s …", self.weights_path
        )
        self._model = load_model(self.config_path, self.weights_path)
        self._model = self._model.to(self.device)
        logger.info("Grounding DINO model loaded on %s.", self.device)

    def detect(
        self,
        image_source: "str | Path | np.ndarray",
        text_prompt: str,
    ) -> list[DetectionResult]:
        """Run open-vocabulary detection on *image_source*.

        Parameters
        ----------
        image_source:
            Path to an image file **or** an ``np.ndarray`` in BGR or RGB
            format (H×W×3, uint8).
        text_prompt:
            Dot-separated list of categories to detect, e.g.
            ``"person . car . red building"``.

        Returns
        -------
        list[DetectionResult]
            Detected objects sorted by descending confidence.
        """
        if not _GDINO_AVAILABLE or self._model is None:
            logger.info("[STUB] detect('%s') → empty list", text_prompt)
            return []

        # Load image
        if isinstance(image_source, (str, Path)):
            image_path = str(image_source)
            image_array, image_tensor = load_image(image_path)
        else:
            # numpy array passed directly
            pil_img = PILImage.fromarray(image_source)
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                pil_img.save(tmp.name)
                tmp_path = tmp.name
            image_array, image_tensor = load_image(tmp_path)
            os.unlink(tmp_path)

        image_tensor = image_tensor.to(self.device)

        boxes, logits, phrases = predict(
            model=self._model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        # Convert from normalised [cx, cy, w, h] → absolute [x1, y1, x2, y2]
        h, w, _ = image_array.shape
        results: list[DetectionResult] = []
        for box, logit, phrase in zip(boxes.tolist(), logits.tolist(), phrases):
            cx, cy, bw, bh = box
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            results.append(
                DetectionResult(
                    label=phrase,
                    confidence=float(logit),
                    box=[x1, y1, x2, y2],
                )
            )

        results.sort(key=lambda r: r.confidence, reverse=True)
        logger.info(
            "Detected %d object(s) for prompt '%s'", len(results), text_prompt
        )
        return results

    def summarise(self, results: list[DetectionResult]) -> str:
        """Return a human-readable summary of detection results.

        The summary is fed back to the LLM so it can reason about what the
        UAV camera currently sees.

        Example output::

            Detected 2 object(s):
              1. person (confidence=0.87, center=(320.0, 240.0))
              2. red car (confidence=0.72, center=(120.5, 180.3))
        """
        if not results:
            return "No objects detected."
        lines = [f"Detected {len(results)} object(s):"]
        for i, r in enumerate(results, 1):
            cx, cy = r.center
            lines.append(
                f"  {i}. {r.label} "
                f"(confidence={r.confidence:.2f}, center=({cx:.1f}, {cy:.1f}))"
            )
        return "\n".join(lines)
