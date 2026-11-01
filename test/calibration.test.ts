import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { describe, it } from "node:test";
import { parseAnswer } from "../src/arms.ts";
import { brier, confidence, coverageAt, reliability } from "../src/calibration.ts";
import { captureFromCsv } from "../src/capture.ts";
import type { Transport } from "../src/cloudflare.ts";
import { build, readClefJsonl } from "../src/dataset.ts";
import { evaluate, parseArm, renderMarkdown } from "../src/evaluate.ts";
import { answerFromSystemOne } from "../src/schema.ts";
import type { Questions } from "../src/types.ts";
import { FAKE_NAMES, humanRule, QUESTIONS } from "../scripts/synthetic.ts";

const close = (a: number, b: number, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

describe("calibration math", () => {
  it("computes pTop and margin", () => {
    const c = confidence({ accept: 0.6, skip: 0.3, needs_info: 0.1 });
    assert.equal(c.top, "accept");
    close(c.pTop, 0.6);
    close(c.margin, 0.3);
  });

  it("computes the multi-class Brier score by hand", () => {
    const opts = ["accept", "skip", "needs_info"];
    close(brier({ accept: 0.6, skip: 0.3, needs_info: 0.1 }, "accept", opts), 0.16 + 0.09 + 0.01);
    close(brier({ accept: 0.6, skip: 0.3, needs_info: 0.1 }, "skip", opts), 0.36 + 0.49 + 0.01);
    close(brier({ accept: 1 }, "skip", opts), 2);
  });

  it("bins pTop into 10 bins and computes ECE", () => {
    const points = [
      { pTop: 0.95, correct: true },
      { pTop: 0.95, correct: false },
      { pTop: 0.55, correct: true },
      { pTop: null, correct: false },
    ];
    const r = reliability(points);
    assert.equal(r.bins.length, 10);
    assert.equal(r.bins[9].n, 2);
    close(r.bins[9].accuracy!, 0.5);
    assert.equal(r.bins[5].n, 1);
    close(r.ece!, (2 * Math.abs(0.95 - 0.5) + 1 * Math.abs(0.55 - 1)) / 3);
    assert.equal(reliability([{ pTop: 1, correct: true }]).bins[9].n, 1);
  });

  it("reports coverage against accuracy, never covering failed calls", () => {
    const points = [
      { pTop: 0.9, correct: true },
      { pTop: 0.8, correct: false },
      { pTop: 0.4, correct: false },
      { pTop: null, correct: false },
    ];
    assert.deepEqual(coverageAt(points, 0.75), { threshold: 0.75, coverage: 0.5, covered: 2, accuracy: 0.5 });
    assert.deepEqual(coverageAt(points, 0), { threshold: 0, coverage: 0.75, covered: 3, accuracy: 1 / 3 });
    assert.equal(coverageAt(points, 0.99).accuracy, null);
  });
});

describe("answerFromSystemOne", () => {
  const qs: Questions = {
    team: { type: "choice", instructions: "Which team?", criteria: { billing: "b", tech: "t", sales: "s" } },
    urgent: { type: "noul", instructions: "Urgent?" },
    severity: { type: "score", instructions: "How bad?", criteria: ["low", "mid", "high"] },
  };
  it("keeps every option's probability, filling unlisted options with 0", () => {
    const a = answerFromSystemOne(
      {
        answers: {
          team: { type: "choice", choice: "tech", probabilities: { tech: 0.7, billing: 0.3 } },
          urgent: { type: "noul", noul: 0.25 },
          severity: { type: "score", score: 1.2, probabilities: { "0": 0.1, "1": 0.6, "2": 0.3 } },
        },
      },
      qs,
    );
    assert.deepEqual(a.decision, { team: "tech", urgent: "false", severity: "1" });
    assert.deepEqual(a.probs.team, { billing: 0.3, tech: 0.7, sales: 0 });
    assert.deepEqual(a.probs.urgent, { true: 0.25, false: 0.75 });
    assert.deepEqual(a.probs.severity, { "0": 0.1, "1": 0.6, "2": 0.3 });
  });

  it("returns no probabilities for text arms", () => {
    const r = parseAnswer({ kind: "chat", model: "openai/gpt-5.5" }, { choices: [{ message: { content: "skip" } }] }, QUESTIONS);
    assert.deepEqual(r, { decision: { decision: "skip" } });
  });
});

describe("evaluate with calibration", () => {
  async function split() {
    const records = captureFromCsv(await readFile("fixtures/leads.csv", "utf8"), QUESTIONS).records;
    const out = build(records, { heldOut: 60, template: "mistral", clefModel: "@cf/cloudflare/clef-flash", redact: { names: FAKE_NAMES } });
    return { train: readClefJsonl(out.files["clef/train.jsonl"]), heldOut: readClefJsonl(out.files["clef/heldout.jsonl"]) };
  }

  const transport: Transport = {
    async send(call) {
      if (call.transport === "workers-ai" && "questions" in call.input) {
        const state = call.input.state as unknown as Parameters<typeof humanRule>[0];
        const choice = humanRule(state);
        const confident = state.distance_miles < 20;
        const top = confident ? 0.9 : 0.5;
        const rest = (1 - top) / 2;
        const probabilities = { accept: rest, skip: rest, needs_info: rest, [choice]: top };
        return { answers: { decision: { type: "choice", choice, probabilities } } };
      }
      if (call.transport === "chat") return { choices: [{ message: { content: "I would accept this one" } }] };
      throw new Error("503 upstream");
    },
  };

  it("reports Brier, reliability and coverage for System One arms, and keeps format failures apart", async () => {
    const { heldOut, train } = await split();
    const arms = [parseArm("clef=systemone:@cf/cloudflare/clef-flash"), parseArm("gpt=chat:openai/gpt-5.5"), parseArm("lora=lora:@cf/mistral/mistral-7b-instruct-v0.2-lora|x|mistral"), parseArm("m=majority")];
    const rep = await evaluate(heldOut, arms, { transport, train, runId: "cal", abstainBelow: 0.8 });
    const by = Object.fromEntries(rep.arms.map((a) => [a.name, a]));
    const cal = by.clef.calibration.decision!;
    assert.equal(cal.n, 60);
    assert.ok(cal.brier > 0 && cal.brier < 0.5, `brier ${cal.brier}`);
    const confident = heldOut.filter((r) => (r.state as { distance_miles: number }).distance_miles < 20).length;
    assert.equal(cal.abstain!.covered, confident);
    assert.equal(cal.abstain!.threshold, 0.8);
    assert.ok(cal.coverage.some((p) => p.threshold === 0.8));
    assert.equal(cal.reliability.reduce((s, b) => s + b.n, 0), 60);
    assert.equal(by.gpt.calibration.decision, null);
    assert.equal(by.m.calibration.decision, null);
    assert.deepEqual(by.gpt.failures, { call: 0, format: 60 });
    assert.deepEqual(by.lora.failures, { call: 60, format: 0 });
    const md = renderMarkdown(rep);
    assert.match(md, /## Calibration `decision`/);
    assert.match(md, /Coverage at pTop >= 0.8/);
    assert.match(md, /\| gpt \| chat \| 0\.0% \| 0\/60 \| \d+ \| \d+ \| 0 \| 60 \|/);
  });
});

describe("anchoring check", () => {
  it("splits agreement by whether the person saw the suggestion", async () => {
    const mk = (i: number, saw: boolean, label: string, logged: string) => ({
      id: `r${i}`,
      createdAt: `2026-10-01T00:00:0${i}.000Z`,
      state: { i },
      questions: QUESTIONS,
      final: { decision: label },
      modelOutput: { decision: logged },
      decidedBy: "human" as const,
      decidedAt: `2026-10-01T00:01:0${i}.000Z`,
      sawSuggestion: saw,
      source: "dataset" as const,
    });
    const records = [mk(0, true, "skip", "skip"), mk(1, true, "accept", "accept"), mk(2, false, "skip", "accept"), mk(3, false, "accept", "accept")];
    const rep = await evaluate(records, [parseArm("logged=replay")], { runId: "a" });
    assert.deepEqual(rep.arms[0].bySuggestion, { saw: { n: 2, exactMatch: 1 }, unseen: { n: 2, exactMatch: 0.5 } });
    assert.match(renderMarkdown(rep), /\| logged \| 100\.0% of 2 \| 50\.0% of 2 \|/);
    const allSaw = await evaluate(records.map((r) => ({ ...r, sawSuggestion: true })), [parseArm("logged=replay")], { runId: "b" });
    assert.equal(allSaw.arms[0].bySuggestion, null);
  });
});

