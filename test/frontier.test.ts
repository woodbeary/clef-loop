import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { buildCall, decisionResponseFormat, parseResult } from "../src/arms.ts";
import { evaluate, parseArm, renderMarkdown } from "../src/evaluate.ts";
import type { Transport } from "../src/cloudflare.ts";
import type { DecisionRecord, Questions } from "../src/types.ts";
import { QUESTIONS } from "../scripts/synthetic.ts";

const STATE = { service: "garage cleanout", distance_miles: 8.5 };

describe("frontier arm with structured output", () => {
  it("asks for a JSON schema limited to the allowed options by default", () => {
    const call = buildCall({ kind: "chat", model: "openai/gpt-5.5" }, STATE, QUESTIONS);
    assert.ok(call.transport === "chat");
    if (call.transport !== "chat") return;
    assert.deepEqual(call.body.response_format, {
      type: "json_schema",
      json_schema: {
        name: "decision",
        strict: true,
        schema: {
          type: "object",
          properties: { decision: { type: "string", enum: ["accept", "skip", "needs_info"] } },
          required: ["decision"],
          additionalProperties: false,
        },
      },
    });
    assert.match(JSON.stringify(call.body.messages), /JSON object that maps each question id/);
  });

  it("covers every question type", () => {
    const qs: Questions = {
      urgent: { type: "noul", instructions: "Urgent?" },
      severity: { type: "score", instructions: "How bad?", criteria: ["low", "high"] },
    };
    const f = decisionResponseFormat(qs) as { json_schema: { schema: { properties: Record<string, { enum: string[] }> } } };
    assert.deepEqual(f.json_schema.schema.properties.urgent.enum, ["true", "false"]);
    assert.deepEqual(f.json_schema.schema.properties.severity.enum, ["0", "1"]);
  });

  it("keeps the bare-text prompt when asked with |text", () => {
    const arm = parseArm("gpt=chat:openai/gpt-5.5|text");
    assert.deepEqual(arm.spec, { kind: "chat", model: "openai/gpt-5.5", format: "text" });
    const call = buildCall(arm.spec as { kind: "chat"; model: string; format: "text" }, STATE, QUESTIONS);
    assert.ok(call.transport === "chat" && !("response_format" in call.body));
    assert.throws(() => parseArm("gpt=chat:openai/gpt-5.5|xml"), /unknown chat format/);
  });

  it("reads a structured answer", () => {
    const raw = { choices: [{ message: { content: '{"decision":"needs_info"}' } }] };
    assert.deepEqual(parseResult({ kind: "chat", model: "openai/gpt-5.5" }, raw, QUESTIONS), { decision: "needs_info" });
  });
});

describe("report says what each arm is", () => {
  it("labels a self-hosted tuned arm as evaluated offline, and names the chat format", async () => {
    const records: DecisionRecord[] = [
      { id: "a", createdAt: "2026-10-01T00:00:00.000Z", state: STATE, questions: QUESTIONS, final: { decision: "accept" }, source: "dataset" },
    ];
    const transport: Transport = {
      async send(call) {
        if (call.transport === "url") return { answers: { decision: { type: "choice", choice: "accept", probabilities: { accept: 1 } } } };
        return { choices: [{ message: { content: '{"decision":"accept"}' } }] };
      },
    };
    const rep = await evaluate(records, [parseArm("tuned=systemone-url:http://127.0.0.1:1/v1/systemone"), parseArm("gpt=chat:openai/gpt-5.5")], { transport, runId: "r" });
    const md = renderMarkdown(rep);
    assert.match(md, /`tuned` is served from a self-hosted System One server: evaluated offline, not a Workers AI or BYOM deployment/);
    assert.match(md, /`gpt` was asked for structured output/);
    assert.equal(rep.arms[1].exactMatch, 1);
  });
});
