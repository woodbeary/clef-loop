import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { decisionFromSystemOne, optionsFor, validateDecision, validateQuestions } from "../src/schema.ts";
import type { Questions } from "../src/types.ts";

// The example from the Clef-flash model page on developers.cloudflare.com.
const DOCS_QUESTIONS: Questions = {
  urgent: { type: "noul", instructions: "Is this support request urgent?" },
  team: {
    type: "choice",
    instructions: "Which team should handle this request?",
    criteria: { billing: "Payments, invoices, and refunds", technical: "Outages, errors, and configuration", sales: "Plans and upgrades" },
  },
  severity: { type: "score", instructions: "How severe is the customer impact?", criteria: ["No impact", "Minor", "Major", "Critical"] },
};

// A response shaped like schema-output.json.
const DOCS_RESPONSE = {
  model: "clef-flash",
  answers: {
    urgent: { type: "noul", noul: 0.93 },
    team: { type: "choice", choice: "technical", probabilities: { billing: 0.03, technical: 0.95, sales: 0.02 }, confidence: 0.9 },
    severity: {
      type: "score",
      score: 2.6,
      legend: { "0": "No impact", "1": "Minor", "2": "Major", "3": "Critical" },
      probabilities: { "0": 0.01, "1": 0.04, "2": 0.3, "3": 0.65 },
      confidence: 0.6,
    },
  },
  usage: { input_tokens: 120, output_tokens: 0 },
};

describe("validateQuestions", () => {
  it("accepts the documented example", () => {
    assert.deepEqual(validateQuestions(DOCS_QUESTIONS), []);
  });

  it("enforces the published limits", () => {
    assert.match(validateQuestions({})[0], /1-64/);
    const tooMany = Object.fromEntries(Array.from({ length: 65 }, (_, i) => [`q${i}`, { type: "noul", instructions: "x" }]));
    assert.match(validateQuestions(tooMany)[0], /1-64/);
    assert.match(validateQuestions({ "bad id": { type: "noul", instructions: "x" } })[0], /must match/);
    assert.match(validateQuestions({ a: { type: "choice", instructions: "x", criteria: { only: "1" } } })[0], /2-255/);
    assert.match(validateQuestions({ a: { type: "score", instructions: "x", criteria: ["one"] } })[0], /2-10/);
    assert.match(validateQuestions({ a: { type: "noul", instructions: "  " } })[0], /instructions/);
    assert.match(validateQuestions({ a: { type: "maybe", instructions: "x" } })[0], /noul, choice or score/);
  });
});

describe("decisionFromSystemOne", () => {
  it("reads choice, noul and score answers", () => {
    assert.deepEqual(decisionFromSystemOne(DOCS_RESPONSE, DOCS_QUESTIONS), { urgent: "true", team: "technical", severity: "3" });
  });

  it("uses the most likely level for score, not the rounded weighted mean", () => {
    const r = structuredClone(DOCS_RESPONSE);
    r.answers.severity.probabilities = { "0": 0.45, "1": 0.0, "2": 0.0, "3": 0.55 };
    r.answers.severity.score = 1.65;
    assert.equal(decisionFromSystemOne(r, DOCS_QUESTIONS).severity, "3");
  });

  it("rejects answers that don't fit the questions", () => {
    const missing = structuredClone(DOCS_RESPONSE) as { answers: Record<string, unknown> };
    delete missing.answers.team;
    assert.throws(() => decisionFromSystemOne(missing, DOCS_QUESTIONS), /no answer for "team"/);

    const unknown = structuredClone(DOCS_RESPONSE);
    unknown.answers.team.choice = "legal";
    assert.throws(() => decisionFromSystemOne(unknown, DOCS_QUESTIONS), /not one of/);

    const badSum = structuredClone(DOCS_RESPONSE);
    badSum.answers.team.probabilities = { billing: 0.5, technical: 0.9, sales: 0.0 };
    assert.throws(() => decisionFromSystemOne(badSum, DOCS_QUESTIONS), /sum to/);

    assert.throws(() => decisionFromSystemOne({ result: "nope" }, DOCS_QUESTIONS), /no answers/);
  });
});

describe("options and decisions", () => {
  it("lists options per type", () => {
    assert.deepEqual(optionsFor(DOCS_QUESTIONS.team), ["billing", "technical", "sales"]);
    assert.deepEqual(optionsFor(DOCS_QUESTIONS.urgent), ["true", "false"]);
    assert.deepEqual(optionsFor(DOCS_QUESTIONS.severity), ["0", "1", "2", "3"]);
  });

  it("validates decisions", () => {
    assert.deepEqual(validateDecision({ urgent: "false", team: "sales", severity: "0" }, DOCS_QUESTIONS), []);
    assert.equal(validateDecision({ urgent: "false", team: "sales" }, DOCS_QUESTIONS).length, 1);
    assert.equal(validateDecision({ urgent: "false", team: "sales", severity: "0", extra: "x" }, DOCS_QUESTIONS).length, 1);
  });
});
