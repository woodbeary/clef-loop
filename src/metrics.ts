// Step 4's metric. Every arm is scored the same way on the same held-out records:
// agreement with the human's final decision, per question. A failed call or an
// answer that can't be parsed counts as wrong; it is never dropped from the
// denominator.

export interface ClassStats {
  support: number;
  predicted: number;
  precision: number | null;
  recall: number | null;
  f1: number;
}

export interface QuestionScore {
  n: number;
  answered: number;
  correct: number;
  accuracy: number;
  /** 95% Wilson score interval for accuracy. */
  ci95: [number, number];
  macroF1: number;
  perClass: Record<string, ClassStats>;
  /** confusion[label][prediction]; an unanswered record is predicted as "(none)". */
  confusion: Record<string, Record<string, number>>;
}

export function scoreQuestion(labels: string[], predictions: Array<string | null>, options: string[]): QuestionScore {
  if (labels.length !== predictions.length) throw new Error("labels and predictions differ in length");
  const n = labels.length;
  let correct = 0;
  let answered = 0;
  const confusion: Record<string, Record<string, number>> = {};
  for (let i = 0; i < n; i++) {
    const p = predictions[i];
    if (p !== null) answered++;
    if (p === labels[i]) correct++;
    confusion[labels[i]] ??= {};
    const key = p ?? "(none)";
    confusion[labels[i]][key] = (confusion[labels[i]][key] ?? 0) + 1;
  }
  const perClass: Record<string, ClassStats> = {};
  for (const o of options) {
    const support = labels.filter((l) => l === o).length;
    const predicted = predictions.filter((p) => p === o).length;
    const tp = labels.filter((l, i) => l === o && predictions[i] === o).length;
    const precision = predicted ? tp / predicted : null;
    const recall = support ? tp / support : null;
    const f1 = precision && recall ? (2 * precision * recall) / (precision + recall) : 0;
    perClass[o] = { support, predicted, precision, recall, f1 };
  }
  const present = options.filter((o) => perClass[o].support > 0 || perClass[o].predicted > 0);
  const macroF1 = present.length ? present.reduce((s, o) => s + perClass[o].f1, 0) / present.length : 0;
  const accuracy = n ? correct / n : 0;
  return { n, answered, correct, accuracy, ci95: wilson(correct, n), macroF1, perClass, confusion };
}

/** Wilson score interval, z = 1.96. */
export function wilson(successes: number, n: number, z = 1.96): [number, number] {
  if (n === 0) return [0, 0];
  const p = successes / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

export interface McNemar {
  /** A right, B wrong. */
  aOnly: number;
  /** A wrong, B right. */
  bOnly: number;
  /** Exact two-sided p-value (binomial on the discordant pairs). */
  p: number;
}

/** Paired comparison of two arms on the same records. */
export function mcnemar(aCorrect: boolean[], bCorrect: boolean[]): McNemar {
  if (aCorrect.length !== bCorrect.length) throw new Error("arms were scored on different records");
  let aOnly = 0;
  let bOnly = 0;
  for (let i = 0; i < aCorrect.length; i++) {
    if (aCorrect[i] && !bCorrect[i]) aOnly++;
    if (!aCorrect[i] && bCorrect[i]) bOnly++;
  }
  const n = aOnly + bOnly;
  if (n === 0) return { aOnly, bOnly, p: 1 };
  const k = Math.min(aOnly, bOnly);
  let tail = 0;
  for (let i = 0; i <= k; i++) tail += Math.exp(logChoose(n, i) + n * Math.log(0.5));
  return { aOnly, bOnly, p: Math.min(1, 2 * tail) };
}

function logChoose(n: number, k: number): number {
  let s = 0;
  for (let i = 1; i <= k; i++) s += Math.log((n - k + i) / i);
  return s;
}

/** Nearest-rank percentile. */
export function percentile(values: number[], p: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.ceil((p / 100) * sorted.length);
  return sorted[Math.min(sorted.length - 1, Math.max(0, rank - 1))];
}
