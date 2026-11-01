"""Supervised fine-tuning of Clef-flash on clef-loop's datasets.

    python train.py --train data/v1/clef/train.jsonl --heldout data/v1/clef/heldout.jsonl \\
        --model-dir weights/base --out ../byom/weights/tuned --dry-run

What it trains: the joint schema head (joint_head.safetensors) and rank-r low-rank
adapters on the Qwen backbone's attention projections. The backbone itself stays frozen.
The objective is label-smoothed cross-entropy plus a Brier term on the option
probabilities, so the tune is pushed toward calibrated probabilities, not just the right
top option. This is supervised training on human and validator labels. It is not RL:
that needs policy-gradient updates driven by the validator's reward with a reference
penalty, which this file doesn't do.

--dry-run needs no GPU and no torch: it checks the data and the time split, reports the
majority baseline the tune must beat on the held-out set, and, given --model-dir, reads
the safetensors headers to say exactly what would be trained and what stays frozen.

The full run needs torch on a GPU (the model card was tested on one H200 with torch 2.11
and transformers 5.10.2) and the release code's joint_schema_model.py. That file ships an
inference API (load_release_model, systemone); the differentiable entry point a trainer
needs isn't documented, so clef_scorer() below must be written against it once read. The
full run has never been executed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import Example, check_split, label_balance, load, options_for  # noqa: E402
from losses import majority_baseline  # noqa: E402
from plan import DEFAULT_TARGETS, make_plan, read_header  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--model-dir", help="the Clef-flash snapshot (safetensors, joint_head.safetensors, joint_schema_model.py)")
    p.add_argument("--out", default=str(HERE.parent / "byom" / "weights" / "tuned"))
    p.add_argument("--rank", type=int, default=16, help="adapter rank (Cloudflare used 256 for Clef itself; small data wants less)")
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--brier-weight", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--validation", type=float, default=0.1, help="newest share of the training data used for early stopping; the held-out set is never used to choose anything")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def describe(train: list[Example], heldout: list[Example], args: argparse.Namespace) -> list[str]:
    lines = ["clef-loop trainer: supervised fine-tuning (cross-entropy + Brier). Not RL."]
    lines.append(f"train: {len(train)} examples, held-out: {len(heldout)} examples")
    who = Counter(e.decided_by for e in train + heldout)
    saw = Counter(e.saw_suggestion for e in train + heldout if e.decided_by == "human")
    lines.append(f"labels by: {dict(who)}; humans who saw the model's suggestion first: {saw.get(True, 0)} of {sum(saw.values())}")
    for qid, counts in label_balance(train).items():
        lines.append(f"train labels for {qid}: {dict(sorted(counts.items()))}")
    for qid, q in train[0].request["questions"].items():
        opts = options_for(q)
        base = majority_baseline(
            np.array([e.targets[qid] for e in train]), np.array([e.targets[qid] for e in heldout]), len(opts)
        )
        lines.append(f"majority baseline on held-out {qid}: accuracy {base['accuracy']:.3f}, Brier {base['brier']:.3f} (a tune must beat this)")
    n_val = max(1, round(len(train) * args.validation))
    steps = args.epochs * -(-(len(train) - n_val) // args.batch_size)
    lines.append(
        f"plan: rank {args.rank}, alpha {args.alpha}, label smoothing {args.label_smoothing}, Brier weight {args.brier_weight}, "
        f"{args.epochs} epochs x batch {args.batch_size} = {steps} steps; early stopping on the newest {n_val} training examples"
    )
    if args.model_dir:
        plan = make_plan(Path(args.model_dir), args.rank, tuple(t.strip() for t in args.targets.split(",") if t.strip()))
        lines.extend(plan.summary().splitlines())
    else:
        lines.append("no --model-dir: pass the Clef-flash snapshot to see exactly what is trained and frozen")
    lines.append(
        f"would write: {args.out}/joint_head.safetensors (replaces the base head), {args.out}/merged_adapters.safetensors "
        "(the adapted projections with the adapters merged in, to fold into the backbone shards before serving) and train_report.json"
    )
    lines.append("then: serve the result with SGLang's /v1/systemone and score it with clef-loop eval --arm tuned=systemone-url:<url>")
    return lines


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    train, heldout = load(args.train), load(args.heldout)
    problems = check_split(train, heldout)
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        return 1
    for line in describe(train, heldout, args):
        print(line)
    if args.dry_run:
        print("dry run: nothing trained")
        return 0
    try:
        import torch  # noqa: F401
    except ImportError:
        print("error: a full run needs torch on a GPU box (see trainer/requirements.txt); use --dry-run here", file=sys.stderr)
        return 2
    if not args.model_dir:
        print("error: --model-dir is required for a full run", file=sys.stderr)
        return 2
    return run(train, args)


def torch_objective(logits: list[Any], targets: list[int], smoothing: float, brier_weight: float) -> Any:
    """The torch twin of losses.objective, summed over questions per example and averaged."""
    import torch

    total = 0
    for z, t in zip(logits, targets):
        logp = torch.log_softmax(z, dim=-1)
        n = z.shape[-1]
        q = torch.full_like(logp, smoothing / max(1, n - 1))
        q[t] = 1 - smoothing
        y = torch.zeros_like(logp)
        y[t] = 1
        total = total + (-(q * logp).sum()) + brier_weight * ((logp.exp() - y) ** 2).sum()
    return total / len(logits)


def clef_scorer(model: Any, processor: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Returns request -> {question id: logits over options_for() order}, with gradients.

    The release code documents load_release_model() and systemone() for inference only.
    Read joint_schema_model.py, then return the head's per-option scores before the
    softmax, computed without torch.no_grad(). Until then this refuses to guess.
    """
    fn = getattr(model, "option_logits", None)
    if fn is None:
        raise NotImplementedError(
            "joint_schema_model.py has no documented differentiable scoring entry point; "
            "implement clef_scorer() against it (see the docstring)"
        )
    return lambda request: fn(processor, request)


def attach_lora(model: Any, targets: set[str], rank: int, alpha: float) -> list[Any]:
    import torch

    class LoRALinear(torch.nn.Module):
        def __init__(self, base: torch.nn.Linear):
            super().__init__()
            self.base = base
            self.a = torch.nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
            self.b = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.a, a=5**0.5)
            self.scale = alpha / rank

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base(x) + ((x.float() @ self.a.T) @ self.b.T * self.scale).to(x.dtype)

        def merged(self) -> torch.Tensor:
            return (self.base.weight.float() + (self.b @ self.a) * self.scale).to(self.base.weight.dtype)

    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, torch.nn.Linear):
                lora = LoRALinear(child)
                setattr(module, child_name, lora)
                wrapped.append((f"{name}.{child_name}" if name else child_name, lora))
    return wrapped


def run(train: list[Example], args: argparse.Namespace) -> int:
    import torch
    from safetensors.torch import save_file

    model_dir = Path(args.model_dir)
    sys.path.insert(0, str(model_dir))
    from joint_schema_model import load_release_model  # shipped with the weights

    model, processor = load_release_model(str(model_dir), device="cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    head_names = set(read_header(model_dir / "joint_head.safetensors"))
    is_head = lambda n: any(n == h or n.endswith(f".{h}") for h in head_names)  # noqa: E731
    head = [p for n, p in model.named_parameters() if is_head(n)]
    if not head:
        raise RuntimeError("no model parameter matches a tensor in joint_head.safetensors; check how load_release_model names the head")
    for p in head:
        p.requires_grad_(True)
    wrapped = attach_lora(model, set(args.targets.split(",")), args.rank, args.alpha)
    params = head + [p for _, m in wrapped for p in (m.a, m.b)]
    opt = torch.optim.AdamW(params, lr=args.lr)
    score = clef_scorer(model, processor)

    ordered = sorted(train, key=lambda e: e.created_at)
    n_val = max(1, round(len(ordered) * args.validation))
    fit, val = ordered[:-n_val], ordered[-n_val:]

    def loss_on(batch: list[Example]) -> Any:
        logits, targets = [], []
        for e in batch:
            out = score(e.request)
            for qid in e.request["questions"]:
                logits.append(out[qid])
                targets.append(e.targets[qid])
        return torch_objective(logits, targets, args.label_smoothing, args.brier_weight)

    best, best_state, history = float("inf"), None, []
    for epoch in range(args.epochs):
        model.train()
        for i in range(0, len(fit), args.batch_size):
            opt.zero_grad()
            loss = loss_on(fit[i : i + args.batch_size])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_on(val))
        history.append({"epoch": epoch + 1, "validation_objective": v})
        if v < best:
            best, best_state = v, [p.detach().clone() for p in params]
    if best_state is not None:
        with torch.no_grad():
            for p, b in zip(params, best_state):
                p.copy_(b)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if is_head(n)}, str(out / "joint_head.safetensors"))
    merged = {f"{name}.weight": m.merged().detach().cpu().contiguous() for name, m in wrapped}
    save_file(merged, str(out / "merged_adapters.safetensors"))
    (out / "train_report.json").write_text(json.dumps({"objective": "cross-entropy + Brier (supervised)", "history": history, "args": vars(args)}, indent=2))
    print(f"wrote {out}; merge merged_adapters.safetensors into the backbone shards before serving")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
