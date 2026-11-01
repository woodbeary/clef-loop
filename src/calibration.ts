// Calibration and abstention. Clef returns a probability for every allowed option,
// and Cloudflare trained it with a Brier loss to calibrate those probabilities, so
// clef-loop measures calibration on your own held-out decisions instead of throwing
// the probabilities away.
//
// - pTop: the probability of the most likely option. margin: pTop minus the runner-up.
// - Brier score: sum over options of (p - y)^2, where y is 1 for the label and 0
//   otherwise. 0 is perfect; 2 is confidently wrong. Averaged over records.
// - Reliability: records binned by pTop into 10 equal bins; in a calibrated model each
//   bin's accuracy matches its mean pTop. ECE is the count-weighted gap.
// - Abstention: an answer with pTop below a threshold goes to a person. Coverage is the
//   share of all records the model still decides; failed calls are never covered.

export interface Confidence {
  top: string;
  pTop: number;
  margin: number;
}

export function confidence(probs: Record<string, number>): Confidence {
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  if (sorted.length === 0) throw new Error("no probabilities");
  const [top, pTop] = sorted[0];
  return { top, pTop, margin: pTop - (sorted[1]?.[1] ?? 0) };
}

export function brier(probs: Record<string, number>, label: string, options: string[]): number {
  let s = 0;
  for (const o of options) s += ((probs[o] ?? 0) - (o === label ? 1 : 0)) ** 2;
  return s;
}

export interface ScoredPoint {
  /** Null when the call failed or the arm returns no probabilities. */
  pTop: number | null;
  correct: boolean;
}

export interface ReliabilityBin {
  lo: number;
  hi: number;
  n: number;
  meanConfidence: number | null;
  accuracy: number | null;
}

export function reliability(points: ScoredPoint[], bins = 10): { bins: ReliabilityBin[]; ece: number | null } {
  const out: ReliabilityBin[] = Array.from({ length: bins }, (_, i) => ({ lo: i / bins, hi: (i + 1) / bins, n: 0, meanConfidence: null, accuracy: null }));
  const sums = out.map(() => ({ conf: 0, right: 0 }));
  let n = 0;
  for (const p of points) {
    if (p.pTop === null) continue;
    const i = Math.min(bins - 1, Math.floor(p.pTop * bins));
    out[i].n++;
    sums[i].conf += p.pTop;
    sums[i].right += p.correct ? 1 : 0;
    n++;
  }
  let gap = 0;
  out.forEach((b, i) => {
    if (b.n === 0) return;
    b.meanConfidence = sums[i].conf / b.n;
    b.accuracy = sums[i].right / b.n;
    gap += b.n * Math.abs(b.meanConfidence - b.accuracy);
  });
  return { bins: out, ece: n ? gap / n : null };
}

export interface CoveragePoint {
  threshold: number;
  /** Records the model still decides, over all records. */
  coverage: number;
  covered: number;
  /** Accuracy on the covered records only. Null when nothing is covered. */
  accuracy: number | null;
}

export function coverageAt(points: ScoredPoint[], threshold: number): CoveragePoint {
  const kept = points.filter((p) => p.pTop !== null && p.pTop >= threshold);
  const right = kept.filter((p) => p.correct).length;
  return {
    threshold,
    coverage: points.length ? kept.length / points.length : 0,
    covered: kept.length,
    accuracy: kept.length ? right / kept.length : null,
  };
}

export const DEFAULT_THRESHOLDS = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
