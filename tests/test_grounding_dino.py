"""Tests for Grounding DINO perception module."""

import pytest
from src.perception.grounding_dino import DetectionResult, GroundingDINODetector


class TestDetectionResult:
    def test_basic_construction(self):
        r = DetectionResult("person", 0.92, [10.0, 20.0, 100.0, 200.0])
        assert r.label == "person"
        assert r.confidence == pytest.approx(0.92)
        assert r.box == [10.0, 20.0, 100.0, 200.0]

    def test_center(self):
        r = DetectionResult("car", 0.8, [0.0, 0.0, 100.0, 80.0])
        cx, cy = r.center
        assert cx == pytest.approx(50.0)
        assert cy == pytest.approx(40.0)

    def test_area(self):
        r = DetectionResult("car", 0.8, [0.0, 0.0, 100.0, 80.0])
        assert r.area == pytest.approx(8000.0)

    def test_invalid_box_raises(self):
        with pytest.raises(ValueError, match="exactly 4 elements"):
            DetectionResult("person", 0.9, [10.0, 20.0])

    def test_to_dict(self):
        r = DetectionResult("building", 0.75, [10.0, 20.0, 110.0, 220.0])
        d = r.to_dict()
        assert d["label"] == "building"
        assert d["confidence"] == pytest.approx(0.75, abs=1e-4)
        assert len(d["box"]) == 4
        assert len(d["center"]) == 2
        assert "area" in d

    def test_repr(self):
        r = DetectionResult("truck", 0.66, [0.0, 0.0, 50.0, 50.0])
        assert "truck" in repr(r)
        assert "0.660" in repr(r)


class TestGroundingDINODetector:
    def test_instantiation(self):
        d = GroundingDINODetector()
        assert d.box_threshold == pytest.approx(0.35)
        assert d.text_threshold == pytest.approx(0.25)

    def test_load_stub(self):
        """load() should not raise even when groundingdino is not installed."""
        d = GroundingDINODetector()
        d.load()  # stub – should be a no-op

    def test_detect_returns_empty_list_in_stub(self, tmp_path):
        """detect() in stub mode (no model) should return an empty list."""
        d = GroundingDINODetector()
        # No model loaded → stub mode
        results = d.detect(str(tmp_path / "dummy.jpg"), "person . car")
        assert isinstance(results, list)
        assert results == []

    def test_summarise_empty(self):
        d = GroundingDINODetector()
        summary = d.summarise([])
        assert summary == "No objects detected."

    def test_summarise_single(self):
        d = GroundingDINODetector()
        r = DetectionResult("person", 0.9, [10.0, 20.0, 100.0, 200.0])
        summary = d.summarise([r])
        assert "1 object" in summary
        assert "person" in summary
        assert "0.90" in summary

    def test_summarise_multiple(self):
        d = GroundingDINODetector()
        results = [
            DetectionResult("person", 0.9, [10.0, 20.0, 100.0, 200.0]),
            DetectionResult("car", 0.7, [200.0, 150.0, 400.0, 300.0]),
        ]
        summary = d.summarise(results)
        assert "2 object" in summary
        assert "person" in summary
        assert "car" in summary

    def test_custom_thresholds(self):
        d = GroundingDINODetector(box_threshold=0.5, text_threshold=0.4)
        assert d.box_threshold == pytest.approx(0.5)
        assert d.text_threshold == pytest.approx(0.4)
