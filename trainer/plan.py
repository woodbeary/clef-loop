"""What gets trained and what stays frozen, worked out from the weights' safetensors
headers alone, so it runs on a laptop with no GPU and no torch.

Clef-flash is a frozen Qwen backbone plus a joint schema head shipped as
joint_head.safetensors. The plan trains every head tensor and adds low-rank adapters
(LoRA) to the backbone's attention projections; everything else stays frozen.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def read_header(path: Path) -> dict[str, dict]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100_000_000:
            raise ValueError(f"{path.name}: safetensors header is implausibly large")
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


@dataclass
class Plan:
    head_tensors: int = 0
    head_params: int = 0
    adapted: list[str] = field(default_factory=list)
    adapter_params: int = 0
    frozen_params: int = 0
    rank: int = 0

    @property
    def trainable_params(self) -> int:
        return self.head_params + self.adapter_params

    def summary(self) -> str:
        total = self.frozen_params + self.head_params
        share = 100 * self.trainable_params / total if total else 0.0
        return (
            f"head: {self.head_tensors} tensors, {self.head_params:,} parameters (trained)\n"
            f"adapters: rank {self.rank} on {len(self.adapted)} projections, {self.adapter_params:,} parameters (trained)\n"
            f"backbone: {self.frozen_params:,} parameters (frozen)\n"
            f"trainable: {self.trainable_params:,} ({share:.2f}% of the released model)"
        )


def _numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def make_plan(model_dir: Path, rank: int, targets: tuple[str, ...] = DEFAULT_TARGETS, head_file: str = "joint_head.safetensors") -> Plan:
    if rank < 1:
        raise ValueError("rank must be at least 1")
    files = sorted(model_dir.glob("*.safetensors"))
    if not any(f.name == head_file for f in files):
        raise FileNotFoundError(f"{model_dir} has no {head_file}; point --model-dir at the Clef-flash snapshot")
    plan = Plan(rank=rank)
    for f in files:
        for name, t in read_header(f).items():
            n = _numel(t["shape"])
            if f.name == head_file:
                plan.head_tensors += 1
                plan.head_params += n
                continue
            plan.frozen_params += n
            leaf = name.rsplit(".", 2)
            if len(t["shape"]) == 2 and name.endswith(".weight") and len(leaf) >= 2 and leaf[-2] in targets:
                out_f, in_f = t["shape"]
                plan.adapted.append(name)
                plan.adapter_params += rank * (in_f + out_f)
    return plan
