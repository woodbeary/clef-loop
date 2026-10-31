"""Cog predictor that serves Clef-flash, base or tuned, with the System One API.

Input and output are the System One request and response bodies as JSON strings,
so a Worker can pass them through unchanged. How Workers AI maps a request onto a
Cog model's inputs isn't public yet; adjust `predict`'s signature to whatever
Cloudflare specifies when you get BYOM access.

Model loading follows the Clef-flash model card (huggingface.co/Cloudflare/clef-flash):
`load_release_model(path)` and `systemone(model, processor, request)` from the
`joint_schema_model.py` that ships with the weights.

Weights:
  weights/base/   optional local copy of the base snapshot (otherwise downloaded)
  weights/tuned/  your tuned files; each one replaces the file of the same name
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

try:
    from cog import BasePredictor, Input
except ImportError:  # Lets the unit tests import this file without Cog installed.
    BasePredictor = object  # type: ignore[assignment,misc]

    def Input(**_kwargs: Any) -> Any:  # type: ignore[no-redef]  # noqa: N802
        return None


HERE = Path(__file__).resolve().parent
BASE_REPO = os.environ.get("CLEF_BASE_REPO", "Cloudflare/clef-flash")
BASE_REVISION = os.environ.get("CLEF_BASE_REVISION")  # pin a commit for reproducible builds
QUESTION_ID = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


def merged_model_dir(base: Path, tuned: Path, out: Path) -> Path:
    """Builds a directory of symlinks: every base file, with tuned files replacing same-named ones."""
    out.mkdir(parents=True, exist_ok=True)
    for src_dir in (base, tuned):
        for src in src_dir.iterdir():
            if src.is_file():
                dst = out / src.name
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(src.resolve())
    return out


def validate_request(body: Any) -> list[str]:
    """The same limits as the published Clef input schema (schema-input.json)."""
    if not isinstance(body, dict):
        return ["request must be a JSON object"]
    problems: list[str] = []
    if "state" not in body:
        problems.append("state is required")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        return problems + ["questions must be an object with 1-64 entries"]
    for qid, q in questions.items():
        if not QUESTION_ID.match(qid):
            problems.append(f"question id {qid!r} is invalid")
        if not isinstance(q, dict):
            problems.append(f"{qid}: must be an object")
            continue
        kind, criteria = q.get("type"), q.get("criteria")
        if kind == "choice" and not (isinstance(criteria, dict) and 2 <= len(criteria) <= 255):
            problems.append(f"{qid}: choice needs 2-255 options")
        elif kind == "score" and not (isinstance(criteria, list) and 2 <= len(criteria) <= 10):
            problems.append(f"{qid}: score needs 2-10 levels")
        elif kind not in ("choice", "score", "noul"):
            problems.append(f"{qid}: type must be noul, choice or score")
    return problems


class Predictor(BasePredictor):
    def setup(self) -> None:
        local_base = HERE / "weights" / "base"
        if local_base.is_dir():
            base = local_base
        else:
            from huggingface_hub import snapshot_download

            base = Path(snapshot_download(BASE_REPO, revision=BASE_REVISION))
        tuned = HERE / "weights" / "tuned"
        path = merged_model_dir(base, tuned, Path("/tmp/clef-model")) if tuned.is_dir() else base
        sys.path.insert(0, str(path))
        from joint_schema_model import load_release_model, systemone  # shipped with the weights

        self.model, self.processor = load_release_model(str(path), device="cuda")
        self.systemone: Callable[[Any, Any, dict], dict] = systemone

    def predict(self, request: str = Input(description="System One request body, as JSON")) -> str:
        body = json.loads(request)
        problems = validate_request(body)
        if problems:
            raise ValueError("; ".join(problems))
        return json.dumps(self.systemone(self.model, self.processor, body))
