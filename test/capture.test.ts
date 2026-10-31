import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { describe, it } from "node:test";
import { captureFromCsv, captureFromGateway, parseDecision, systemOneInput, unwrapEnvelope } from "../src/capture.ts";
import { QUESTIONS } from "../scripts/synthetic.ts";

describe("captureFromCsv", () => {
  it("reads the synthetic fixture", async () => {
    const csv = await readFile("fixtures/leads.csv", "utf8");
    const { records, skipped } = captureFromCsv(csv, QUESTIONS);
    assert.equal(skipped.length, 0);
    assert.equal(records.length, 303);
    const first = records[0];
    assert.equal(first.id, "lead-0001");
    assert.equal(typeof first.state, "object");
    assert.ok(first.final.decision);
    assert.ok(first.modelOutput?.decision);
  });

  it("skips bad rows with a reason instead of failing the whole file", () => {
    const csv = [
      "id,created_at,state,final,model_output",
      'ok,2026-09-01T00:00:00Z,"{""a"":1}",accept,skip',
      "bad-date,yesterday,text,accept,",
      "bad-label,2026-09-01T00:00:00Z,text,maybe,",
      'json-label,2026-09-02T00:00:00Z,plain text,"{""decision"":""skip""}",',
    ].join("\n");
    const { records, skipped } = captureFromCsv(csv, QUESTIONS);
    assert.deepEqual(records.map((r) => r.id), ["ok", "json-label"]);
    assert.deepEqual(records[0].state, { a: 1 });
    assert.equal(records[1].state, "plain text");
    assert.deepEqual(skipped.map((s) => s.id), ["bad-date", "bad-label"]);
    assert.match(skipped[1].reason, /not one of/);
  });

  it("requires the core columns", () => {
    assert.throws(() => captureFromCsv("id,state\n1,x\n", QUESTIONS), /missing columns: created_at, final/);
  });
});

describe("parseDecision", () => {
  it("accepts a bare option only when there is one question", () => {
    assert.deepEqual(parseDecision("skip", QUESTIONS, "final"), { decision: "skip" });
    const two = { ...QUESTIONS, urgent: { type: "noul" as const, instructions: "Urgent?" } };
    assert.throws(() => parseDecision("skip", two, "final"), /JSON object/);
    assert.deepEqual(parseDecision('{"decision":"skip","urgent":"true"}', two, "final"), { decision: "skip", urgent: "true" });
  });
});

describe("captureFromGateway", () => {
  // A fake of the documented AI Gateway logs API: list (paged, per_page <= 50),
  // then /request and /response per log.
  function fakeApi() {
    const logs = Array.from({ length: 53 }, (_, i) => ({
      id: `log-${i}`,
      created_at: new Date(Date.UTC(2026, 8, 1, i)).toISOString(),
      provider: "workers-ai",
      model: "@cf/cloudflare/clef-flash",
      metadata: i === 1 ? JSON.stringify({ final: '{"decision":"skip"}', clef_loop_eval: true }) // eval traffic
        : i === 2 ? JSON.stringify({ route: "base" }) // no human label yet
        : JSON.stringify({ final: JSON.stringify({ decision: i % 2 ? "accept" : "skip" }), route: "base" }),
    }));
    const calls: string[] = [];
    const f = (async (url: string, init?: RequestInit) => {
      calls.push(url);
      assert.equal((init?.headers as Record<string, string>).Authorization, "Bearer t0ken");
      const u = new URL(url);
      const m = u.pathname.match(/\/logs\/([^/]+)\/(request|response)$/);
      if (m) {
        const i = Number(m[1].split("-")[1]);
        if (m[2] === "request") {
          // Alternate between the bare body and the /ai/run envelope.
          const body = { model: "clef-flash", state: { n: i }, questions: QUESTIONS };
          return Response.json(i % 3 === 0 ? { success: true, result: { model: "@cf/cloudflare/clef-flash", input: body } } : body);
        }
        const answers = { decision: { type: "choice", choice: "accept", probabilities: { accept: 0.8, skip: 0.1, needs_info: 0.1 }, confidence: 0.7 } };
        return Response.json({ success: true, result: JSON.stringify({ model: "clef-flash", answers, usage: { input_tokens: 1, output_tokens: 0 } }) });
      }
      const page = Number(u.searchParams.get("page"));
      const per = Number(u.searchParams.get("per_page"));
      assert.ok(per <= 50);
      assert.equal(u.searchParams.get("order_by"), "created_at");
      return Response.json({ success: true, result: logs.slice((page - 1) * per, page * per), result_info: { total_count: logs.length } });
    }) as typeof fetch;
    return { f, calls };
  }

  it("pages through logs, skips eval traffic and unlabeled logs, and reads both body shapes", async () => {
    const { f, calls } = fakeApi();
    const { records, skipped } = await captureFromGateway({ accountId: "acct", gatewayId: "gw", apiToken: "t0ken", fetch: f });
    assert.equal(records.length, 51);
    assert.equal(skipped.length, 0);
    assert.ok(!records.some((r) => r.id === "log-1" || r.id === "log-2"));
    assert.deepEqual(records[0].state, { n: 0 });
    assert.deepEqual(records[0].final, { decision: "skip" });
    assert.deepEqual(records[0].modelOutput, { decision: "accept" });
    assert.equal(records[0].source, "ai-gateway");
    assert.equal(calls.filter((c) => c.includes("page=")).length, 2);
  });

  it("stops at --limit", async () => {
    const { f } = fakeApi();
    const { records } = await captureFromGateway({ accountId: "acct", gatewayId: "gw", apiToken: "t0ken", fetch: f, limit: 5 });
    assert.equal(records.length, 5);
  });

  it("reports API errors without the account id", async () => {
    const f = (async () => new Response("nope", { status: 403 })) as unknown as typeof fetch;
    await assert.rejects(
      captureFromGateway({ accountId: "secret-account", gatewayId: "gw", apiToken: "t", fetch: f }),
      (e: Error) => e.message.includes("403") && !e.message.includes("secret-account"),
    );
  });
});

describe("log body helpers", () => {
  it("unwraps the API envelope and JSON strings", () => {
    assert.deepEqual(unwrapEnvelope({ success: true, result: '{"a":1}' }), { a: 1 });
    assert.deepEqual(unwrapEnvelope({ a: 1 }), { a: 1 });
  });

  it("finds state and questions in either request shape", () => {
    assert.deepEqual(systemOneInput({ input: { state: 1, questions: {} } }), { state: 1, questions: {} });
    assert.deepEqual(systemOneInput({ state: 1, questions: {} }), { state: 1, questions: {} });
    assert.throws(() => systemOneInput({ messages: [] }), /no state or questions/);
  });
});
