// Validation against the published Clef / System One schema, and normalization of
// System One answers into one option id per question.
//
// Limits come from schema-input.json and schema-output.json on
// developers.cloudflare.com/workers-ai/models/clef-flash/ (checked 2026-10-31):
// 1-64 questions; ids of letters, digits, '_', '.', '-' up to 100 characters;
// choice has 2-255 options; score has 2-10 levels.

import type { Decision, Json, Question, Questions } from "./types.ts";

const QUESTION_ID = /^[A-Za-z0-9_.-]{1,100}$/;

/** Returns a list of problems; an empty list means the questions are valid. */
export function validateQuestions(questions: unknown): string[] {
  const problems: string[] = [];
  if (!isObject(questions)) return ["questions must be an object"];
  const ids = Object.keys(questions);
  if (ids.length < 1 || ids.length > 64) problems.push(`questions must have 1-64 entries, got ${ids.length}`);
  for (const id of ids) {
    if (!QUESTION_ID.test(id)) problems.push(`question id "${id}" must match ${QUESTION_ID}`);
    const q = questions[id];
    if (!isObject(q)) {
      problems.push(`${id}: must be an object`);
      continue;
    }
    if (isEmptyInstructions(q.instructions)) problems.push(`${id}: instructions are required`);
    if (q.type === "choice") {
      if (!isObject(q.criteria)) problems.push(`${id}: choice criteria must be an object of option id -> description`);
      else {
        const n = Object.keys(q.criteria).length;
        if (n < 2 || n > 255) problems.push(`${id}: choice needs 2-255 options, got ${n}`);
        if (Object.keys(q.criteria).some((o) => o.length === 0)) problems.push(`${id}: option ids must be non-empty`);
      }
    } else if (q.type === "score") {
      if (!Array.isArray(q.criteria)) problems.push(`${id}: score criteria must be an array of levels`);
      else if (q.criteria.length < 2 || q.criteria.length > 10)
        problems.push(`${id}: score needs 2-10 levels, got ${q.criteria.length}`);
    } else if (q.type === "noul") {
      if (q.criteria !== undefined && !isObject(q.criteria)) problems.push(`${id}: noul criteria must be an object`);
    } else {
      problems.push(`${id}: type must be noul, choice or score`);
    }
  }
  return problems;
}

/** The allowed option ids for a question, in a stable order. */
export function optionsFor(question: Question): string[] {
  switch (question.type) {
    case "choice":
      return Object.keys(question.criteria);
    case "noul":
      return ["true", "false"];
    case "score":
      return question.criteria.map((_, i) => String(i));
  }
}

/** Checks that a decision answers every question with an allowed option. */
export function validateDecision(decision: unknown, questions: Questions): string[] {
  if (!isObject(decision)) return ["decision must be an object"];
  const problems: string[] = [];
  for (const [id, q] of Object.entries(questions)) {
    const v = decision[id];
    if (typeof v !== "string") problems.push(`${id}: missing`);
    else if (!optionsFor(q).includes(v)) problems.push(`${id}: "${v}" is not one of ${optionsFor(q).join(", ")}`);
  }
  for (const id of Object.keys(decision)) if (!(id in questions)) problems.push(`${id}: not a question`);
  return problems;
}

/** Per question, the probability the model gave each allowed option. */
export type Probabilities = Record<string, Record<string, number>>;

export interface Answer {
  decision: Decision;
  probs: Probabilities;
}

/**
 * Turns a System One response body ({ model, answers, usage }) into a Decision.
 * Throws with a reason when the answers don't match the questions, so callers can
 * fall back instead of acting on a malformed decision.
 */
export function decisionFromSystemOne(body: unknown, questions: Questions): Decision {
  return answerFromSystemOne(body, questions).decision;
}

/** Like decisionFromSystemOne, and keeps the probabilities for calibration and abstention. */
export function answerFromSystemOne(body: unknown, questions: Questions): Answer {
  if (!isObject(body) || !isObject(body.answers)) throw new Error("response has no answers object");
  const answers = body.answers;
  const decision: Decision = {};
  const probs: Probabilities = {};
  for (const [id, q] of Object.entries(questions)) {
    const a = answers[id];
    if (!isObject(a)) throw new Error(`no answer for "${id}"`);
    const options = optionsFor(q);
    if (q.type === "choice") {
      if (typeof a.choice !== "string") throw new Error(`"${id}": choice answer has no choice`);
      probs[id] = checkProbabilities(id, a.probabilities, options);
      decision[id] = a.choice;
    } else if (q.type === "noul") {
      if (typeof a.noul !== "number" || a.noul < 0 || a.noul > 1) throw new Error(`"${id}": noul must be a probability`);
      probs[id] = { true: a.noul, false: 1 - a.noul };
      decision[id] = a.noul >= 0.5 ? "true" : "false";
    } else {
      probs[id] = checkProbabilities(id, a.probabilities, options);
      // The decision is the most likely level; `score` itself is a weighted mean and
      // can land between levels.
      decision[id] = Object.entries(probs[id]).sort((x, y) => y[1] - x[1])[0][0];
    }
  }
  const problems = validateDecision(decision, questions);
  if (problems.length) throw new Error(problems.join("; "));
  return { decision, probs };
}

function checkProbabilities(id: string, probs: unknown, options: string[]): Record<string, number> {
  if (!isObject(probs)) throw new Error(`"${id}": probabilities missing`);
  const out: Record<string, number> = Object.fromEntries(options.map((o) => [o, 0]));
  let sum = 0;
  for (const [k, v] of Object.entries(probs)) {
    if (!options.includes(k)) throw new Error(`"${id}": unknown option "${k}" in probabilities`);
    if (typeof v !== "number" || v < 0 || v > 1) throw new Error(`"${id}": probability for "${k}" is not in [0, 1]`);
    out[k] = v;
    sum += v;
  }
  if (Math.abs(sum - 1) > 0.01) throw new Error(`"${id}": probabilities sum to ${sum.toFixed(3)}, not 1`);
  return out;
}

export function isObject(v: unknown): v is Record<string, Json> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isEmptyInstructions(v: unknown): boolean {
  if (v === undefined || v === null) return true;
  if (typeof v === "string") return v.trim().length === 0;
  return false;
}
