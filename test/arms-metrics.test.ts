import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { answerText, buildCall, decisionPrompt, parseAnswerText, parseResult, renderInference, renderTraining } from "../src/arms.ts";
import { mcnemar, percentile, scoreQuestion, wilson } from "../src/metrics.ts";
import { QUESTIONS } from "../scripts/synthetic.ts";

const STATE = { service: "garage cleanout", distance_miles: 8.5 };

describe("buildCall", () => {
  it("sends Clef the documented System One body", () => {
    const call = buildCall({ kind: "systemone", model: "@cf/cloudflare/clef-flash" }, STATE, QUESTIONS);
    assert.deepEqual(call, { transport: "workers-ai", model: "@cf/cloudflare/clef-flash", input: { model: "clef-flash", state: STATE, questions: QUESTIONS } });
    const big = buildCall({ kind: "systemone", model: "@cf/cloudflare/clef" }, STATE, QUESTIONS);
    assert.equal(big.transport === "workers-ai" && big.input.model, "clef");
    const omni = buildCall({ kind: "systemone", model: "@cf/cloudflare/clef-omni" }, STATE, QUESTIONS);
    assert.equal(omni.transport === "workers-ai" && omni.input.model, "clef-omni");
    const byom = buildCall({ kind: "systemone", model: "some-byom-id", selector: "clef-flash" }, STATE, QUESTIONS);
    assert.equal(byom.transport === "workers-ai" && byom.input.model, "clef-flash");
  });

  it("sends a LoRA model a raw templated prompt and the fine-tune name", () => {
    const call = buildCall({ kind: "lora", model: "@cf/mistral/mistral-7b-instruct-v0.2-lora", lora: "leads-v1", template: "mistral" }, STATE, QUESTIONS);
    assert.equal(call.transport, "workers-ai");
    if (call.transport !== "workers-ai") return;
    assert.equal(call.input.lora, "leads-v1");
    assert.equal(call.input.raw, true);
    assert.equal(call.input.temperature, 0);
    assert.equal(call.input.prompt, renderInference("mistral", decisionPrompt(STATE, QUESTIONS)));
    const base = buildCall({ kind: "lora", model: "@cf/mistral/mistral-7b-instruct-v0.2-lora", template: "mistral" }, STATE, QUESTIONS);
    assert.ok(base.transport === "workers-ai" && !("lora" in base.input));
  });

  it("sends a chat model the same prompt", () => {
    const call = buildCall({ kind: "chat", model: "openai/gpt-5.5" }, STATE, QUESTIONS);
    assert.ok(call.transport === "chat");
    if (call.transport !== "chat") return;
    assert.equal(call.body.model, "openai/gpt-5.5");
    assert.match(JSON.stringify(call.body.messages), /needs_info/);
  });

  it("trains on exactly the prompt it infers with", () => {
    const p = decisionPrompt(STATE, QUESTIONS);
    for (const t of ["mistral", "gemma", "llama2"] as const) {
      assert.ok(renderTraining(t, p, "accept").startsWith(renderInference(t, p)));
    }
    assert.equal(answerText({ decision: "skip" }, QUESTIONS), "skip");
  });
});

describe("parsing answers", () => {
  it("accepts an exact option id, with light formatting", () => {
    for (const s of ["accept", " Accept.", '"skip"', "needs_info\nBecause the load is unknown.", "```\nskip\n```"]) {
      assert.ok(parseAnswerText(s, QUESTIONS).decision);
    }
  });

  it("does not guess from prose", () => {
    for (const s of ["I would accept this", "maybe", "", "accept or skip"]) {
      assert.throws(() => parseAnswerText(s, QUESTIONS), /not an option id/);
    }
  });

  it("reads JSON answers and rejects JSON that doesn't fit", () => {
    assert.deepEqual(parseAnswerText('{"decision": "skip"}', QUESTIONS), { decision: "skip" });
    assert.throws(() => parseAnswerText('{"decision": "later"}', QUESTIONS), /does not match/);
  });

  it("reads each transport's result shape", () => {
    assert.deepEqual(parseResult({ kind: "lora", model: "m", template: "mistral" }, { response: "skip" }, QUESTIONS), { decision: "skip" });
    assert.deepEqual(
      parseResult({ kind: "chat", model: "openai/gpt-5.5" }, { choices: [{ message: { role: "assistant", content: "accept" } }] }, QUESTIONS),
      { decision: "accept" },
    );
    assert.throws(() => parseResult({ kind: "chat", model: "x" }, { error: "rate limited" }, QUESTIONS), /no message content/);
  });
});

describe("metrics", () => {
  it("scores a small case computed by hand", () => {
    const labels = ["a", "a", "a", "b", "b", "c"];
    const preds = ["a", "a", "b", "b", null, "a"];
    const s = scoreQuestion(labels, preds, ["a", "b", "c"]);
    assert.equal(s.correct, 3);
    assert.equal(s.answered, 5);
    assert.equal(s.accuracy, 0.5);
    // a: tp 2, predicted 3, support 3 -> P 2/3, R 2/3
    assert.ok(Math.abs(s.perClass.a.precision! - 2 / 3) < 1e-12);
    assert.ok(Math.abs(s.perClass.a.recall! - 2 / 3) < 1e-12);
    // b: tp 1, predicted 2, support 2 -> P 1/2, R 1/2; c: never predicted -> P null, F1 0
    assert.equal(s.perClass.b.f1, 0.5);
    assert.equal(s.perClass.c.precision, null);
    assert.ok(Math.abs(s.macroF1 - (2 / 3 + 0.5 + 0) / 3) < 1e-12);
    assert.deepEqual(s.confusion.b, { b: 1, "(none)": 1 });
  });

  it("computes the Wilson interval", () => {
    const [lo, hi] = wilson(8, 10);
    assert.ok(Math.abs(lo - 0.4902) < 1e-3 && Math.abs(hi - 0.9433) < 1e-3);
    assert.deepEqual(wilson(0, 0), [0, 0]);
  });

  it("computes the exact McNemar test", () => {
    const a = [...Array(10).fill(true), ...Array(2).fill(false), ...Array(20).fill(true)];
    const b = [...Array(10).fill(false), ...Array(2).fill(true), ...Array(20).fill(true)];
    const m = mcnemar(a, b);
    assert.equal(m.aOnly, 10);
    assert.equal(m.bOnly, 2);
    assert.ok(Math.abs(m.p - 0.03857) < 1e-4); // 2 * P(X <= 2), X ~ Bin(12, 0.5) = 2 * 79/4096
    assert.equal(mcnemar([true], [true]).p, 1);
  });

  it("takes nearest-rank percentiles", () => {
    assert.equal(percentile([5, 1, 3, 2, 4], 50), 3);
    assert.equal(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95), 10);
    assert.equal(percentile([], 50), null);
  });
});
