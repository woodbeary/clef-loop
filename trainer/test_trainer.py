"""Tests for the trainer's data checks, objective, freeze plan and dry run. They need
NumPy only; the torch parity test runs when torch is installed."""

import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import data
import losses
import plan

HERE = Path(__file__).resolve().parent
QUESTIONS = {"decision": {"type": "choice", "instructions": "Take it?", "criteria": {"accept": "a", "skip": "s", "needs_info": "n"}}}


def line(i, label="accept", by="human", at=None, saw=False, state=None):
    created = at or f"2026-09-{1 + i // 24:02d}T{i % 24:02d}:00:00.000Z"
    o = {
        "id": f"r{i}",
        "created_at": created,
        "request": {"model": "clef-flash", "state": state if state is not None else {"n": i}, "questions": QUESTIONS},
        "label": {"decision": label},
        "decided_by": by,
        "decided_at": created,
        "saw_suggestion": saw,
    }
    return json.dumps(o)


def write(tmp, name, lines):
    p = Path(tmp) / name
    p.write_text("\n".join(lines) + "\n")
    return p


def safetensors_header(path, tensors):
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = math.prod(shape) * plan.DTYPE_BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


class Data(unittest.TestCase):
    def test_reads_targets_in_option_order(self):
        e = data.parse_line(line(0, "needs_info", saw=True), 1)
        self.assertEqual(e.targets, {"decision": 2})
        self.assertTrue(e.saw_suggestion)

    def test_refuses_agent_and_unattributed_labels(self):
        with self.assertRaisesRegex(ValueError, "agent labels are refused"):
            data.parse_line(line(0, by="agent"), 1)
        o = json.loads(line(0))
        del o["decided_by"]
        with self.assertRaisesRegex(ValueError, "decided_by"):
            data.parse_line(json.dumps(o), 1)
        with self.assertRaisesRegex(ValueError, "not one of"):
            data.parse_line(line(0, "maybe"), 1)

    def test_split_must_hold_out_the_newest_and_not_leak(self):
        train = [data.parse_line(line(i), i) for i in range(5)]
        heldout = [data.parse_line(line(i), i) for i in range(5, 8)]
        self.assertEqual(data.check_split(train, heldout), [])
        self.assertTrue(any("newer" in p for p in data.check_split(heldout, train)))
        leak = [data.parse_line(line(9, state={"n": 1}), 9)]
        self.assertTrue(any("also appear" in p for p in data.check_split(train, leak)))


class Objective(unittest.TestCase):
    def test_cross_entropy_and_brier_by_hand(self):
        logits = np.array([[math.log(3), 0.0]])
        target = np.array([0])
        self.assertAlmostEqual(losses.cross_entropy(logits, target), -math.log(0.75))
        self.assertAlmostEqual(losses.cross_entropy(logits, target, 0.1), -(0.9 * math.log(0.75) + 0.1 * math.log(0.25)))
        self.assertAlmostEqual(losses.brier(logits, target), 0.25**2 + 0.25**2)
        self.assertAlmostEqual(losses.objective(logits, target, 0.0, 2.0), -math.log(0.75) + 2 * 0.125)

    def test_smoothing_spreads_over_the_other_options(self):
        q = losses.smoothed_targets(3, np.array([1]), 0.1)
        np.testing.assert_allclose(q, [[0.05, 0.9, 0.05]])
        with self.assertRaises(ValueError):
            losses.smoothed_targets(3, np.array([0]), 1.0)

    def test_majority_baseline(self):
        b = losses.majority_baseline(np.array([0, 0, 0, 1]), np.array([0, 1]), 3)
        self.assertAlmostEqual(b["accuracy"], 0.5)
        self.assertAlmostEqual(b["brier"], ((0.75 - 1) ** 2 + 0.25**2 + (0.75**2 + 0.75**2)) / 2, places=6)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch is not installed")
    def test_torch_objective_matches_numpy(self):
        import torch

        import train

        z = np.array([[0.2, 1.5, -0.3], [2.0, 0.1, 0.0]])
        t = np.array([1, 0])
        want = losses.objective(z, t, 0.05, 1.0)
        got = float(train.torch_objective([torch.tensor(r) for r in z], list(t), 0.05, 1.0))
        self.assertAlmostEqual(got, want, places=6)


class FreezePlan(unittest.TestCase):
    def test_trains_the_head_and_adapters_and_freezes_the_backbone(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            safetensors_header(d / "joint_head.safetensors", {"head.proj.weight": ("BF16", [32, 16]), "head.proj.bias": ("BF16", [32])})
            safetensors_header(
                d / "model-00001-of-00001.safetensors",
                {
                    "model.embed_tokens.weight": ("BF16", [100, 16]),
                    "model.layers.0.self_attn.q_proj.weight": ("BF16", [16, 16]),
                    "model.layers.0.self_attn.k_proj.weight": ("BF16", [4, 16]),
                    "model.layers.0.self_attn.q_proj.bias": ("BF16", [16]),
                    "model.layers.0.mlp.up_proj.weight": ("BF16", [64, 16]),
                },
            )
            p = plan.make_plan(d, rank=2)
            self.assertEqual(p.head_tensors, 2)
            self.assertEqual(p.head_params, 32 * 16 + 32)
            self.assertEqual(p.adapted, ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight"])
            self.assertEqual(p.adapter_params, 2 * (16 + 16) + 2 * (16 + 4))
            self.assertEqual(p.frozen_params, 100 * 16 + 16 * 16 + 4 * 16 + 16 + 64 * 16)
            self.assertIn("(frozen)", p.summary())

    def test_needs_the_head_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "joint_head.safetensors"):
                plan.make_plan(Path(tmp), rank=2)


class DryRun(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-I", str(HERE / "train.py"), *args], capture_output=True, text=True)

    def test_dry_run_reports_the_plan_and_trains_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = ["accept", "skip", "accept", "needs_info", "accept", "skip", "accept", "accept"]
            train = write(tmp, "train.jsonl", [line(i, labels[i], saw=i % 2 == 0) for i in range(6)])
            held = write(tmp, "heldout.jsonl", [line(i, labels[i], by="validator") for i in range(6, 8)])
            d = Path(tmp) / "model"
            d.mkdir()
            safetensors_header(d / "joint_head.safetensors", {"head.w": ("F32", [3, 8])})
            safetensors_header(d / "model.safetensors", {"model.layers.0.self_attn.v_proj.weight": ("F32", [8, 8])})
            r = self.run_cli("--train", str(train), "--heldout", str(held), "--model-dir", str(d), "--dry-run", "--out", str(Path(tmp) / "out"))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("supervised fine-tuning (cross-entropy + Brier). Not RL.", r.stdout)
            self.assertIn("labels by: {'human': 6, 'validator': 2}", r.stdout)
            self.assertIn("humans who saw the model's suggestion first: 3 of 6", r.stdout)
            self.assertIn("majority baseline on held-out decision: accuracy 1.000", r.stdout)
            self.assertIn("adapters: rank 16 on 1 projections", r.stdout)
            self.assertIn("dry run: nothing trained", r.stdout)
            self.assertFalse((Path(tmp) / "out").exists())

    def test_refuses_a_bad_split_and_a_full_run_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = write(tmp, "train.jsonl", [line(i) for i in range(5, 8)])
            held = write(tmp, "heldout.jsonl", [line(i) for i in range(3)])
            r = self.run_cli("--train", str(train), "--heldout", str(held), "--dry-run")
            self.assertEqual(r.returncode, 1)
            self.assertIn("hold out the newest", r.stderr)
            if __import__("importlib").util.find_spec("torch") is None:
                r = self.run_cli("--train", str(held), "--heldout", str(train))
                self.assertEqual(r.returncode, 2)
                self.assertIn("needs torch on a GPU box", r.stderr)


if __name__ == "__main__":
    unittest.main()
