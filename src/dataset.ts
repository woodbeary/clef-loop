// Steps 2 and 3: redact, split, and write the datasets.
//
// Outputs (all derived from the same redacted split):
//   clef/train.jsonl, clef/heldout.jsonl
//     One System One request per line plus the human's label:
//     {"id", "created_at", "request": {"model", "state", "questions"}, "label": {...}}
//     `request` is exactly the documented Clef input (schema-input.json), so it can
//     be replayed against Clef, a BYOM deployment or a self-hosted SGLang server.
//     Cloudflare has not published an upload format for Clef fine-tuning (the RL
//     service is run with its FDE team), so `label` is clef-loop's own field.
//   lora/train.csv
//     The format in Cloudflare's AutoTrain tutorial: one column named `text`, each
//     row the base model's chat template with the prompt and the answer.
//   lora/heldout.jsonl
//     The same held-out records, for evaluating the LoRA arm.
//   manifest.json
//     Counts, label balance, time ranges, redaction counts and file hashes.

import { answerText, decisionPrompt, defaultSelector, renderTraining } from "./arms.ts";
import { toCsv } from "./csv.ts";
import { auditResidual, redactJson, type RedactOptions, type RedactionCounts, type ResidualFinding } from "./redact.ts";
import { labelCounts, split, type SplitResult } from "./split.ts";
import type { DecisionRecord, TemplateName } from "./types.ts";

export interface BuildOptions {
  heldOut: number;
  template: TemplateName;
  /** The Clef model the requests target. */
  clefModel: string;
  redact: RedactOptions;
  /** Write even if the residual PII audit finds something. Off by default. */
  allowResidual?: boolean;
}

export interface BuildOutput {
  files: Record<string, string>;
  split: SplitResult;
  redactions: RedactionCounts;
  residual: ResidualFinding[];
}

export function redactRecords(
  records: DecisionRecord[],
  opts: RedactOptions,
): { records: DecisionRecord[]; counts: RedactionCounts; residual: ResidualFinding[] } {
  const counts: RedactionCounts = {};
  const residual: ResidualFinding[] = [];
  const out = records.map((r) => {
    const state = redactJson(r.state, opts, counts);
    for (const f of auditResidual(state)) residual.push({ ...f, path: `${r.id}:${f.path}` });
    return { ...r, state };
  });
  return { records: out, counts, residual };
}

export function build(records: DecisionRecord[], opts: BuildOptions): BuildOutput {
  const red = redactRecords(records, opts.redact);
  if (red.residual.length && !opts.allowResidual) {
    const sample = red.residual
      .slice(0, 5)
      .map((f) => `${f.path} (${f.kind}: ${f.sample})`)
      .join("; ");
    throw new Error(
      `PII audit found ${red.residual.length} item(s) that still look like PII after redaction: ${sample}. ` +
        "Fix the data or add names/keys to the redaction options, then rebuild.",
    );
  }
  const s = split(red.records, { heldOut: opts.heldOut });
  const selector = defaultSelector(opts.clefModel);
  const clefLine = (r: DecisionRecord) =>
    JSON.stringify({
      id: r.id,
      created_at: r.createdAt,
      request: { model: selector, state: r.state, questions: r.questions },
      label: r.final,
      ...(r.modelOutput ? { logged_output: r.modelOutput } : {}),
    });
  const files: Record<string, string> = {
    "clef/train.jsonl": s.train.map(clefLine).join("\n") + "\n",
    "clef/heldout.jsonl": s.heldOut.map(clefLine).join("\n") + "\n",
    "lora/train.csv": toCsv(
      ["text"],
      s.train.map((r) => [renderTraining(opts.template, decisionPrompt(r.state, r.questions), answerText(r.final, r.questions))]),
    ),
    "lora/heldout.jsonl": s.heldOut.map(clefLine).join("\n") + "\n",
  };
  const range = (rs: DecisionRecord[]) => (rs.length ? { from: rs[0].createdAt, to: rs[rs.length - 1].createdAt } : null);
  const manifest = {
    tool: "clef-loop",
    version: "0.1.0",
    clef_model: opts.clefModel,
    lora_template: opts.template,
    counts: {
      input: records.length,
      duplicates_removed: s.duplicatesRemoved,
      label_conflicts: s.conflicts.length,
      train: s.train.length,
      heldout: s.heldOut.length,
    },
    time: { train: range(s.train), heldout: range(s.heldOut), cutoff: s.cutoff },
    labels: { train: labelCounts(s.train), heldout: labelCounts(s.heldOut) },
    redactions: red.counts,
    residual_pii_findings: red.residual.length,
    sources: [...new Set(records.map((r) => r.source))],
  };
  files["manifest.json"] = JSON.stringify(manifest, null, 2) + "\n";
  return { files, split: s, redactions: red.counts, residual: red.residual };
}

/** Reads clef/heldout.jsonl (or clef/train.jsonl) back into records. */
export function readClefJsonl(text: string): DecisionRecord[] {
  return text
    .split("\n")
    .filter((l) => l.trim() !== "")
    .map((line, i) => {
      const o = JSON.parse(line);
      if (!o.request?.questions || !o.label) throw new Error(`line ${i + 1}: missing request.questions or label`);
      return {
        id: String(o.id),
        createdAt: String(o.created_at),
        state: o.request.state,
        questions: o.request.questions,
        final: o.label,
        ...(o.logged_output ? { modelOutput: o.logged_output } : {}),
        source: "dataset" as const,
      };
    });
}
