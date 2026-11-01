// Step 4: score every arm on the same held-out records with the same metric.

import { buildCall, FormatError, parseAnswer, type CallableArm } from "./arms.ts";
import { brier, confidence, coverageAt, DEFAULT_THRESHOLDS, reliability, type CoveragePoint, type ReliabilityBin } from "./calibration.ts";
import type { Transport } from "./cloudflare.ts";
import { mcnemar, percentile, scoreQuestion, type McNemar, type QuestionScore } from "./metrics.ts";
import { optionsFor } from "./schema.ts";
import { labelCounts } from "./split.ts";
import type { ArmOutcome, ArmSpec, Decision, DecisionRecord } from "./types.ts";

export interface NamedArm {
  name: string;
  spec: ArmSpec;
}

export interface EvalOptions {
  transport?: Transport;
  /** Needed by the `majority` arm. */
  train?: DecisionRecord[];
  concurrency?: number;
  runId: string;
  clock?: () => number;
  /** Route answers whose top probability is below this to a person, and report coverage and accuracy at it. */
  abstainBelow?: number;
}

export interface CalibrationReport {
  /** Records with probabilities (answered by an arm that returns them). */
  n: number;
  brier: number;
  ece: number | null;
  meanPTop: number;
  meanMargin: number;
  reliability: ReliabilityBin[];
  /** Coverage and accuracy at each threshold; failed calls are never covered. */
  coverage: CoveragePoint[];
  abstain: CoveragePoint | null;
}

export interface ArmReport {
  name: string;
  spec: ArmSpec;
  /** Share of records where every question matches the human. */
  exactMatch: number;
  questions: Record<string, QuestionScore>;
  /** Client wall time per call in ms, network included. Null for arms that make no call. */
  latencyMs: { p50: number | null; p95: number | null } | null;
  errors: Array<{ id: string; error: string; kind: "call" | "format" }>;
  /** Failed calls and answers that weren't an allowed option, counted apart, so a format miss isn't read as a wrong judgment. */
  failures: { call: number; format: number };
  /** Per question; null for arms that return no probabilities. */
  calibration: Record<string, CalibrationReport | null>;
}

export interface EvalReport {
  runId: string;
  n: number;
  heldOut: { from: string | null; to: string | null };
  arms: ArmReport[];
  /** Every pair of arms, on the same records (exact match). */
  pairwise: Array<{ a: string; b: string } & McNemar>;
  outcomes: Record<string, ArmOutcome[]>;
}

export function parseArm(text: string): NamedArm {
  const eq = text.indexOf("=");
  if (eq <= 0) throw new Error(`--arm must look like name=kind:..., got "${text}"`);
  const name = text.slice(0, eq);
  const rest = text.slice(eq + 1);
  const colon = rest.indexOf(":");
  const kind = colon === -1 ? rest : rest.slice(0, colon);
  const arg = colon === -1 ? "" : rest.slice(colon + 1);
  const need = (what: string) => {
    if (!arg) throw new Error(`arm "${name}" (${kind}) needs ${what}`);
    return arg;
  };
  switch (kind) {
    case "replay":
    case "majority":
      return { name, spec: { kind } };
    case "systemone":
      return { name, spec: { kind, model: need("a model id, e.g. systemone:@cf/cloudflare/clef-flash") } };
    case "systemone-url":
      return { name, spec: { kind, url: need("a URL, e.g. systemone-url:http://localhost:30000/v1/systemone") } };
    case "chat": {
      // chat:<model>[|text]: structured output by default; |text asks for the bare answer.
      const [model, format] = need("a model id, e.g. chat:openai/gpt-5.5").split("|");
      if (format !== undefined && format !== "text" && format !== "json_schema") throw new Error(`unknown chat format "${format}"`);
      return { name, spec: { kind, model, ...(format ? { format } : {}) } };
    }
    case "lora": {
      // lora:<model>[|<finetune>][|<template>]
      const [model, lora, template = "mistral"] = need("lora:<model>|<finetune>|<template>").split("|");
      if (!["mistral", "gemma", "llama2"].includes(template)) throw new Error(`unknown template "${template}"`);
      return {
        name,
        spec: { kind, model, template: template as "mistral" | "gemma" | "llama2", ...(lora ? { lora } : {}) },
      };
    }
    default:
      throw new Error(`unknown arm kind "${kind}"`);
  }
}

export async function evaluate(records: DecisionRecord[], arms: NamedArm[], opts: EvalOptions): Promise<EvalReport> {
  if (records.length === 0) throw new Error("no held-out records");
  if (arms.length === 0) throw new Error("no arms");
  const names = new Set<string>();
  for (const a of arms) {
    if (names.has(a.name)) throw new Error(`duplicate arm name "${a.name}"`);
    names.add(a.name);
  }
  const qkey = (r: DecisionRecord) => Object.keys(r.questions).sort().join(",");
  const mixed = records.find((r) => qkey(r) !== qkey(records[0]));
  if (mixed) throw new Error(`record ${mixed.id} has different questions; evaluate one question set at a time`);
  const clock = opts.clock ?? (() => performance.now());
  const outcomes: Record<string, ArmOutcome[]> = {};
  for (const arm of arms) {
    outcomes[arm.name] = await runArm(arm, records, opts, clock);
  }

  const reports = arms.map((arm) => report(arm, records, outcomes[arm.name], opts.abstainBelow));
  const correct = (name: string) =>
    outcomes[name].map((o, i) => o.ok && sameDecision(o.decision, records[i].final, records[i]));
  const pairwise = arms.flatMap((a, i) =>
    arms.slice(i + 1).map((b) => ({ a: a.name, b: b.name, ...mcnemar(correct(a.name), correct(b.name)) })),
  );
  return {
    runId: opts.runId,
    n: records.length,
    heldOut: { from: records[0]?.createdAt ?? null, to: records[records.length - 1]?.createdAt ?? null },
    arms: reports,
    pairwise,
    outcomes,
  };
}

async function runArm(
  arm: NamedArm,
  records: DecisionRecord[],
  opts: EvalOptions,
  clock: () => number,
): Promise<ArmOutcome[]> {
  const spec = arm.spec;
  if (spec.kind === "replay") {
    return records.map((r) =>
      r.modelOutput ? { ok: true, decision: r.modelOutput, ms: 0 } : { ok: false, error: "no logged output", ms: 0 },
    );
  }
  if (spec.kind === "majority") {
    if (!opts.train?.length) throw new Error(`arm "${arm.name}" (majority) needs the training records`);
    const counts = labelCounts(opts.train);
    const mode: Decision = Object.fromEntries(
      Object.entries(counts).map(([q, c]) => [q, Object.entries(c).sort((x, y) => y[1] - x[1] || (x[0] < y[0] ? -1 : 1))[0][0]]),
    );
    return records.map(() => ({ ok: true, decision: mode, ms: 0 }));
  }
  const transport = opts.transport;
  if (!transport) throw new Error(`arm "${arm.name}" makes calls and needs a transport`);
  const callable: CallableArm = spec;
  const out: ArmOutcome[] = new Array(records.length);
  let next = 0;
  const worker = async () => {
    while (next < records.length) {
      const i = next++;
      const r = records[i];
      const start = clock();
      try {
        const raw = await transport.send(buildCall(callable, r.state, r.questions), {
          clef_loop_eval: true,
          run: opts.runId,
          arm: arm.name,
        });
        const answer = parseAnswer(callable, raw, r.questions);
        out[i] = { ok: true, decision: answer.decision, ms: clock() - start, ...(answer.probs ? { probs: answer.probs } : {}) };
      } catch (e) {
        out[i] = { ok: false, error: (e as Error).message, ms: clock() - start, kind: e instanceof FormatError ? "format" : "call" };
      }
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, opts.concurrency ?? 4) }, worker));
  return out;
}

function report(arm: NamedArm, records: DecisionRecord[], outcomes: ArmOutcome[], abstainBelow?: number): ArmReport {
  const questions: Record<string, QuestionScore> = {};
  const calibration: Record<string, CalibrationReport | null> = {};
  for (const [qid, q] of Object.entries(records[0].questions)) {
    const labels = records.map((r) => r.final[qid]);
    const preds = outcomes.map((o) => (o.ok ? (o.decision[qid] ?? null) : null));
    questions[qid] = scoreQuestion(labels, preds, optionsFor(q));
    calibration[qid] = calibrate(qid, optionsFor(q), labels, outcomes, abstainBelow);
  }
  const exact = outcomes.filter((o, i) => o.ok && sameDecision(o.decision, records[i].final, records[i])).length;
  const calls = arm.spec.kind !== "replay" && arm.spec.kind !== "majority";
  const ms = outcomes.map((o) => o.ms);
  const errors = outcomes.flatMap((o, i) => (o.ok ? [] : [{ id: records[i].id, error: o.error, kind: o.kind ?? ("call" as const) }]));
  return {
    name: arm.name,
    spec: arm.spec,
    exactMatch: exact / records.length,
    questions,
    latencyMs: calls ? { p50: percentile(ms, 50), p95: percentile(ms, 95) } : null,
    errors,
    failures: { call: errors.filter((e) => e.kind === "call").length, format: errors.filter((e) => e.kind === "format").length },
    calibration,
  };
}

function calibrate(qid: string, options: string[], labels: string[], outcomes: ArmOutcome[], abstainBelow?: number): CalibrationReport | null {
  if (!outcomes.some((o) => o.ok && o.probs?.[qid])) return null;
  const points = outcomes.map((o, i) => {
    const probs = o.ok ? o.probs?.[qid] : undefined;
    return { pTop: probs ? confidence(probs).pTop : null, correct: o.ok && o.decision[qid] === labels[i] };
  });
  let sumBrier = 0;
  let sumTop = 0;
  let sumMargin = 0;
  let n = 0;
  outcomes.forEach((o, i) => {
    const probs = o.ok ? o.probs?.[qid] : undefined;
    if (!probs) return;
    const c = confidence(probs);
    sumBrier += brier(probs, labels[i], options);
    sumTop += c.pTop;
    sumMargin += c.margin;
    n++;
  });
  const rel = reliability(points);
  const thresholds = [...new Set([...DEFAULT_THRESHOLDS, ...(abstainBelow === undefined ? [] : [abstainBelow])])].sort((a, b) => a - b);
  return {
    n,
    brier: sumBrier / n,
    ece: rel.ece,
    meanPTop: sumTop / n,
    meanMargin: sumMargin / n,
    reliability: rel.bins,
    coverage: thresholds.map((t) => coverageAt(points, t)),
    abstain: abstainBelow === undefined ? null : coverageAt(points, abstainBelow),
  };
}

function sameDecision(a: Decision, b: Decision, r: DecisionRecord): boolean {
  return Object.keys(r.questions).every((q) => a[q] === b[q]);
}

const pct = (x: number) => `${(100 * x).toFixed(1)}%`;
const num = (x: number | null) => (x === null ? "n/a" : x.toFixed(2));

export function renderMarkdown(rep: EvalReport, note?: string): string {
  const lines: string[] = [];
  lines.push(`# clef-loop eval: ${rep.runId}`, "");
  if (note) lines.push(`> ${note}`, "");
  lines.push(
    `${rep.n} held-out records, ${rep.heldOut.from ?? "?"} to ${rep.heldOut.to ?? "?"}. ` +
      "Metric: agreement with the human's final decision. Failed calls and unparseable answers count as wrong.",
    "",
  );
  lines.push(
    "| Arm | Kind | Exact match | Answered | p50 ms | p95 ms | Call failures | Format failures |",
    "|---|---|---|---|---|---|---|---|",
  );
  for (const a of rep.arms) {
    const answered = Math.min(...Object.values(a.questions).map((q) => q.answered));
    lines.push(
      `| ${a.name} | ${a.spec.kind} | ${pct(a.exactMatch)} | ${answered}/${rep.n} | ${a.latencyMs?.p50?.toFixed(0) ?? "n/a"} | ${
        a.latencyMs?.p95?.toFixed(0) ?? "n/a"
      } | ${a.failures.call} | ${a.failures.format} |`,
    );
  }
  lines.push(
    "",
    "Format failures are answers that came back but weren't an allowed option. They count as wrong above; " +
      "the column keeps them apart from wrong judgments.",
  );
  for (const a of rep.arms) {
    if (a.spec.kind === "systemone-url") {
      lines.push(
        "",
        `\`${a.name}\` is served from a self-hosted System One server: evaluated offline, not a Workers AI or BYOM deployment.`,
      );
    }
    if (a.spec.kind === "chat") {
      lines.push(
        "",
        `\`${a.name}\` was asked for ${(a.spec.format ?? "json_schema") === "json_schema" ? "structured output (a JSON schema limited to the allowed options)" : "the bare answer as text"}.`,
      );
    }
  }
  const qids = Object.keys(rep.arms[0].questions);
  for (const qid of qids) {
    lines.push("", `## Question \`${qid}\``, "", "| Arm | Accuracy (95% CI) | Macro-F1 | Per class: precision / recall |", "|---|---|---|---|");
    for (const a of rep.arms) {
      const q = a.questions[qid];
      const cls = Object.entries(q.perClass)
        .map(([o, s]) => `${o} ${num(s.precision)} / ${num(s.recall)}`)
        .join("; ");
      lines.push(`| ${a.name} | ${pct(q.accuracy)} (${pct(q.ci95[0])} to ${pct(q.ci95[1])}) | ${q.macroF1.toFixed(3)} | ${cls} |`);
    }
  }
  for (const qid of qids) {
    const cal = rep.arms.filter((a) => a.calibration[qid]);
    if (!cal.length) continue;
    const abstain = cal.find((a) => a.calibration[qid]!.abstain)?.calibration[qid]!.abstain?.threshold;
    lines.push(
      "",
      `## Calibration \`${qid}\``,
      "",
      "Brier score: sum over options of (p - label)^2, averaged; lower is better. ECE: gap between confidence and accuracy over 10 bins.",
      "",
      `| Arm | With probabilities | Brier | ECE | Mean pTop | Mean margin${abstain === undefined ? "" : ` | Coverage at pTop >= ${abstain} | Accuracy when covered`} |`,
      `|---|---|---|---|---|---${abstain === undefined ? "" : "|---|---"}|`,
    );
    for (const a of cal) {
      const c = a.calibration[qid]!;
      const ab = c.abstain ? ` | ${pct(c.abstain.coverage)} (${c.abstain.covered}/${rep.n}) | ${c.abstain.accuracy === null ? "n/a" : pct(c.abstain.accuracy)}` : "";
      lines.push(`| ${a.name} | ${c.n}/${rep.n} | ${c.brier.toFixed(3)} | ${c.ece === null ? "n/a" : c.ece.toFixed(3)} | ${c.meanPTop.toFixed(3)} | ${c.meanMargin.toFixed(3)}${ab} |`);
    }
    for (const a of cal) {
      const c = a.calibration[qid]!;
      lines.push("", `Reliability, ${a.name}:`, "", "| pTop bin | n | Mean pTop | Accuracy |", "|---|---|---|---|");
      for (const b of c.reliability) {
        if (b.n === 0) continue;
        lines.push(`| ${b.lo.toFixed(1)}-${b.hi.toFixed(1)} | ${b.n} | ${b.meanConfidence!.toFixed(3)} | ${pct(b.accuracy!)} |`);
      }
      lines.push("", `Coverage against accuracy, ${a.name} (below the threshold, a person decides):`, "", "| Threshold | Coverage | Accuracy when covered |", "|---|---|---|");
      for (const p of c.coverage) lines.push(`| ${p.threshold} | ${pct(p.coverage)} (${p.covered}/${rep.n}) | ${p.accuracy === null ? "n/a" : pct(p.accuracy)} |`);
    }
  }
  if (rep.pairwise.length) {
    lines.push(
      "",
      "## Paired comparison (exact McNemar, same records)",
      "",
      "| A | B | A right, B wrong | B right, A wrong | p |",
      "|---|---|---|---|---|",
    );
    for (const p of rep.pairwise) lines.push(`| ${p.a} | ${p.b} | ${p.aOnly} | ${p.bOnly} | ${p.p.toPrecision(3)} |`);
  }
  lines.push(
    "",
    "Latency is client wall time from the machine that ran this eval, network included. It is not production latency; " +
      "see docs/MEASURING.md for how to measure that.",
  );
  return lines.join("\n") + "\n";
}
