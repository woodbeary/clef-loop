// Step 5's routing rules, kept pure so they are tested without a Worker.
//
// - Sticky assignment: the same key always lands in the same bucket for a given
//   route version, so a customer doesn't flip between models mid-conversation.
// - Per-request fallback: if the tuned model errors, times out or returns an
//   answer that doesn't fit the questions, the request is served by the base model.
// - Route-level revert: when the tuned route is shown to fail more often than allowed
//   (see judge), the whole route reverts to base until someone resets it.

import { wilson } from "./metrics.ts";

export type Mode = "off" | "shadow" | "canary";
export type Route = "base" | "tuned";

/** FNV-1a 32-bit hash of salt + key, mapped to 0..9999 (basis points). */
export function bucket(key: string, salt: string): number {
  let h = 0x811c9dc5;
  const s = `${salt}\u0000${key}`;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h % 10_000;
}

/** Which model serves this request. Shadow mode always serves base (tuned runs on the side). */
export function chooseRoute(opts: {
  mode: Mode;
  percent: number;
  key: string;
  salt: string;
  tunedConfigured: boolean;
  reverted: boolean;
}): Route {
  if (opts.mode !== "canary" || !opts.tunedConfigured || opts.reverted) return "base";
  const pct = Math.min(100, Math.max(0, opts.percent));
  return bucket(opts.key, opts.salt) < pct * 100 ? "tuned" : "base";
}

export interface RevertConfig {
  /** How many recent requests per route to look at. */
  window: number;
  /** The failure rate tuned must be shown to exceed (0.01 = 1%), or base's own rate if that is higher. */
  maxFailureRate: number;
  /** Don't judge until at least this many tuned requests are in the window. Base's rate is used once it has as many. */
  minSamples: number;
}

export interface WindowStats {
  n: number;
  failures: number;
  rate: number | null;
  /** 95% Wilson interval for the failure rate. */
  ci95: [number, number] | null;
}

export interface Judgement {
  revert: boolean;
  /** max(maxFailureRate, base's failure rate once base has minSamples). */
  limit: number;
  tuned: WindowStats;
  base: WindowStats;
  reason: string | null;
}

export function windowStats(failed: boolean[]): WindowStats {
  const n = failed.length;
  const failures = failed.filter(Boolean).length;
  return { n, failures, rate: n ? failures / n : null, ci95: n ? wilson(failures, n) : null };
}

/**
 * Reverts only when the evidence says tuned fails more often than allowed: the lower
 * end of the 95% Wilson interval for tuned's failure rate must pass the limit, and the
 * limit is base's own failure rate when that is higher. One transient 429 at request
 * 50 doesn't revert a good canary, and an outage that hits both routes doesn't blame
 * tuned. A failure is an error, a timeout, an answer that doesn't fit the questions,
 * or a validator reject.
 */
export function judge(tunedFailed: boolean[], baseFailed: boolean[], cfg: RevertConfig): Judgement {
  const tuned = windowStats(tunedFailed);
  const base = windowStats(baseFailed);
  const baseRate = base.n >= cfg.minSamples ? base.rate! : 0;
  const limit = Math.max(cfg.maxFailureRate, baseRate);
  if (tuned.n < cfg.minSamples) return { revert: false, limit, tuned, base, reason: null };
  const lower = tuned.ci95![0];
  if (lower <= limit) return { revert: false, limit, tuned, base, reason: null };
  const pctf = (x: number) => `${(100 * x).toFixed(1)}%`;
  const baseNote = base.n >= cfg.minSamples ? `base ${pctf(base.rate!)} of ${base.n}` : `base has ${base.n} of ${cfg.minSamples} samples`;
  return {
    revert: true,
    limit,
    tuned,
    base,
    reason:
      `${tuned.failures} of the last ${tuned.n} tuned requests failed (${pctf(tuned.rate!)}, 95% lower bound ${pctf(lower)}), ` +
      `above the limit ${pctf(limit)} (${baseNote})`,
  };
}

export function parseRevertConfig(env: {
  REVERT_WINDOW?: string;
  REVERT_MIN_SAMPLES?: string;
  REVERT_MAX_FAILURE_RATE?: string;
}): RevertConfig {
  const window = intVar(env.REVERT_WINDOW, 200, "REVERT_WINDOW");
  const minSamples = intVar(env.REVERT_MIN_SAMPLES, 50, "REVERT_MIN_SAMPLES");
  const maxFailureRate = Number(env.REVERT_MAX_FAILURE_RATE ?? "0.01");
  if (!(maxFailureRate >= 0 && maxFailureRate < 1)) throw new Error("REVERT_MAX_FAILURE_RATE must be in [0, 1)");
  if (minSamples > window) throw new Error("REVERT_MIN_SAMPLES must not exceed REVERT_WINDOW");
  return { window, minSamples, maxFailureRate };
}

function intVar(v: string | undefined, dflt: number, name: string): number {
  if (v === undefined || v === "") return dflt;
  const n = Number(v);
  if (!Number.isInteger(n) || n < 1) throw new Error(`${name} must be a positive integer`);
  return n;
}
