"""Tests for the Cog predictor's own logic. They don't load Clef (that needs a GPU and
the weights); they check request validation, the weights overlay and the I/O shape."""

import json
import tempfile
import unittest
from pathlib import Path

import predict


class ValidateRequest(unittest.TestCase):
    def test_accepts_a_documented_request(self):
        body = {
            "model": "clef-flash",
            "state": "Checkout has been failing for every customer for the last hour.",
            "questions": {
                "urgent": {"type": "noul", "instructions": "Is this urgent?"},
                "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "x", "technical": "y"}},
                "severity": {"type": "score", "instructions": "How severe?", "criteria": ["None", "Minor", "Major"]},
            },
        }
        self.assertEqual(predict.validate_request(body), [])

    def test_rejects_bad_shapes(self):
        self.assertIn("state is required", predict.validate_request({"questions": {"a": {"type": "noul"}}}))
        self.assertTrue(predict.validate_request({"state": "x", "questions": {}}))
        bad = {"state": "x", "questions": {"bad id!": {"type": "choice", "criteria": {"only": "one"}}}}
        problems = predict.validate_request(bad)
        self.assertTrue(any("invalid" in p for p in problems))
        self.assertTrue(any("2-255" in p for p in problems))


class MergedModelDir(unittest.TestCase):
    def test_tuned_files_replace_base_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, tuned, out = root / "base", root / "tuned", root / "out"
            base.mkdir()
            tuned.mkdir()
            (base / "config.json").write_text("base-config")
            (base / "joint_head.safetensors").write_text("base-head")
            (tuned / "joint_head.safetensors").write_text("tuned-head")
            predict.merged_model_dir(base, tuned, out)
            self.assertEqual((out / "config.json").read_text(), "base-config")
            self.assertEqual((out / "joint_head.safetensors").read_text(), "tuned-head")


class PredictShape(unittest.TestCase):
    def test_passes_the_request_through_and_returns_json(self):
        p = predict.Predictor()
        p.model, p.processor = object(), object()
        seen = {}

        def fake_systemone(model, processor, body):
            seen["body"] = body
            return {"model": "clef-flash", "answers": {"ok": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 1, "output_tokens": 0}}

        p.systemone = fake_systemone
        req = {"model": "clef-flash", "state": "s", "questions": {"ok": {"type": "noul", "instructions": "ok?"}}}
        out = json.loads(p.predict(json.dumps(req)))
        self.assertEqual(seen["body"], req)
        self.assertEqual(out["answers"]["ok"]["noul"], 0.9)

    def test_rejects_invalid_requests_before_the_model(self):
        p = predict.Predictor()
        p.systemone = lambda *a: self.fail("model should not run")
        with self.assertRaises(ValueError):
            p.predict(json.dumps({"state": "s", "questions": {}}))


if __name__ == "__main__":
    unittest.main()
