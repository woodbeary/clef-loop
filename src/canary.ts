// Step 5's routing rules, kept pure so they are tested without a Worker.
//
// - Sticky assignment: the same key always lands in the same bucket for a given
//   route version, so a customer doesn't flip between models mid-conversation.
// - Per-request fallback: if the tuned model errors, times out or returns an
//   answer that doesn't fit the questions, the request is served by the base model.
// - Route-level revert: if tuned failures exceed a rate over the last N tuned
//   requests, the whole route reverts to base until someone resets it.

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
  /** How many recent tuned requests to look at. */
  window: number;
  /** Revert when failures / window exceeds this (0.01 = 1%). */
  maxFailureRate: number;
  /** Don't judge until at least this many tuned requests are in the window. */
  minSamples: number;
}

export interface RevertState {
  /** Most recent last. true = the tuned request failed and fell back. */
  recent: boolean[];
  reverted: boolean;
  reason: string | null;
  revertedAt: string | null;
}

export const EMPTY_STATE: RevertState = { recent: [], reverted: false, reason: null, revertedAt: null };

export function recordOutcome(state: RevertState, failed: boolean, cfg: RevertConfig, now: Date): RevertState {
  const recent = [...state.recent, failed].slice(-cfg.window);
  if (state.reverted) return { ...state, recent };
  const failures = recent.filter(Boolean).length;
  const rate = failures / recent.length;
  if (recent.length >= cfg.minSamples && rate > cfg.maxFailureRate) {
    return {
      recent,
      reverted: true,
      reason: `${failures} of the last ${recent.length} tuned requests failed (${(100 * rate).toFixed(1)}% > ${(
        100 * cfg.maxFailureRate
      ).toFixed(1)}%)`,
      revertedAt: now.toISOString(),
    };
  }
  return { ...state, recent };
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
