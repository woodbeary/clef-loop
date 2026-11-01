"""Reads clef-loop's dataset files (clef/train.jsonl, clef/heldout.jsonl) for training.

Every line is a System One request plus the label and who made it. A label must come
from a person or a validator: agent labels are refused here too, because training on
a model's own answers rewards it for agreeing with itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Example:
    id: str
    created_at: str
    request: dict[str, Any]
    label: dict[str, str]
    decided_by: str
    saw_suggestion: bool
    targets: dict[str, int] = field(default_factory=dict)


def options_for(question: dict[str, Any]) -> list[str]:
    """The allowed option ids, in the same order as clef-loop's optionsFor()."""
    kind = question.get("type")
    if kind == "choice":
        return list(question["criteria"].keys())
    if kind == "noul":
        return ["true", "false"]
    if kind == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    raise ValueError(f"unknown question type {kind!r}")


def parse_line(line: str, n: int) -> Example:
    o = json.loads(line)
    where = f"line {n}"
    request, label = o.get("request"), o.get("label")
    if not isinstance(request, dict) or not isinstance(request.get("questions"), dict) or not isinstance(label, dict):
        raise ValueError(f"{where}: missing request.questions or label")
    by = o.get("decided_by")
    if by == "agent":
        raise ValueError(f"{where}: agent labels are refused")
    if by not in ("human", "validator"):
        raise ValueError(f"{where}: decided_by must be human or validator (rebuild datasets made before clef-loop 0.2)")
    if not isinstance(o.get("decided_at"), str):
        raise ValueError(f"{where}: decided_at is required")
    targets: dict[str, int] = {}
    for qid, q in request["questions"].items():
        opts = options_for(q)
        if label.get(qid) not in opts:
            raise ValueError(f"{where}: label for {qid!r} is {label.get(qid)!r}, not one of {opts}")
        targets[qid] = opts.index(label[qid])
    return Example(
        id=str(o["id"]),
        created_at=str(o["created_at"]),
        request=request,
        label=label,
        decided_by=by,
        saw_suggestion=bool(o.get("saw_suggestion", False)),
        targets=targets,
    )


def load(path: str | Path) -> list[Example]:
    lines = Path(path).read_text(encoding="utf8").splitlines()
    return [parse_line(line, i + 1) for i, line in enumerate(lines) if line.strip()]


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def check_split(train: list[Example], heldout: list[Example]) -> list[str]:
    """Problems that would make the held-out score meaningless."""
    problems: list[str] = []
    if not train:
        problems.append("no training examples")
    if not heldout:
        problems.append("no held-out examples")
    if train and heldout and max(_t(e.created_at) for e in train) > min(_t(e.created_at) for e in heldout):
        problems.append("training data is newer than held-out data: hold out the newest decisions")
    seen = {json.dumps(e.request, sort_keys=True) for e in train}
    leaked = sum(1 for e in heldout if json.dumps(e.request, sort_keys=True) in seen)
    if leaked:
        problems.append(f"{leaked} held-out request(s) also appear in training")
    qsets = {json.dumps(sorted(e.request["questions"].keys())) for e in train + heldout}
    if len(qsets) > 1:
        problems.append("examples have different question sets; train one question set at a time")
    return problems


def label_balance(examples: list[Example]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for e in examples:
        for qid, v in e.label.items():
            out.setdefault(qid, {})
            out[qid][v] = out[qid].get(v, 0) + 1
    return out
