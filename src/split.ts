// Train / held-out split. The held-out set is the newest records, so the eval asks
// the question production will ask: how does a model trained on the past do on
// what came next? Exact duplicates are removed first so the same input can't land
// on both sides.

import type { DecisionRecord, Json } from "./types.ts";

export interface SplitOptions {
  /** A count (e.g. 400) or a fraction of the deduplicated records (e.g. 0.2). */
  heldOut: number;
}

export interface SplitResult {
  train: DecisionRecord[];
  heldOut: DecisionRecord[];
  /** createdAt of the first held-out record. Everything in train is earlier or equal. */
  cutoff: string | null;
  duplicatesRemoved: number;
  /** Identical inputs that humans labeled differently. Kept once (earliest); reported here. */
  conflicts: Array<{ ids: string[] }>;
}

export function split(records: DecisionRecord[], opts: SplitOptions): SplitResult {
  if (!Number.isFinite(opts.heldOut) || opts.heldOut <= 0) throw new Error(`held-out must be a count or a fraction, got ${opts.heldOut}`);
  const { unique, duplicatesRemoved, conflicts } = dedupe(records);
  const sorted = [...unique].sort((a, b) => compareTime(a, b));
  const n = opts.heldOut >= 1 ? Math.floor(opts.heldOut) : Math.round(sorted.length * opts.heldOut);
  if (n <= 0 || n >= sorted.length) {
    throw new Error(`held-out size ${n} must be between 1 and ${sorted.length - 1} (have ${sorted.length} unique records)`);
  }
  const train = sorted.slice(0, sorted.length - n);
  const heldOut = sorted.slice(sorted.length - n);
  return { train, heldOut, cutoff: heldOut[0]?.createdAt ?? null, duplicatesRemoved, conflicts };
}

export function dedupe(records: DecisionRecord[]): {
  unique: DecisionRecord[];
  duplicatesRemoved: number;
  conflicts: Array<{ ids: string[] }>;
} {
  const byInput = new Map<string, DecisionRecord[]>();
  for (const r of [...records].sort(compareTime)) {
    const key = canonical({ state: r.state, questions: r.questions as unknown as Json });
    const group = byInput.get(key);
    if (group) group.push(r);
    else byInput.set(key, [r]);
  }
  const unique: DecisionRecord[] = [];
  const conflicts: Array<{ ids: string[] }> = [];
  let duplicatesRemoved = 0;
  for (const group of byInput.values()) {
    unique.push(group[0]);
    duplicatesRemoved += group.length - 1;
    const labels = new Set(group.map((r) => canonical(r.final)));
    if (labels.size > 1) conflicts.push({ ids: group.map((r) => r.id) });
  }
  return { unique, duplicatesRemoved, conflicts };
}

/** Label counts per question: { decision: { accept: 12, skip: 30 } }. */
export function labelCounts(records: DecisionRecord[]): Record<string, Record<string, number>> {
  const out: Record<string, Record<string, number>> = {};
  for (const r of records) {
    for (const [q, v] of Object.entries(r.final)) {
      out[q] ??= {};
      out[q][v] = (out[q][v] ?? 0) + 1;
    }
  }
  return out;
}

/** JSON with object keys sorted, so equal values always serialize the same way. */
export function canonical(value: unknown): string {
  return JSON.stringify(sortKeys(value));
}

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value as object)
        .sort()
        .map((k) => [k, sortKeys((value as Record<string, unknown>)[k])]),
    );
  }
  return value;
}

function compareTime(a: DecisionRecord, b: DecisionRecord): number {
  const t = Date.parse(a.createdAt) - Date.parse(b.createdAt);
  return t !== 0 ? t : a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}
