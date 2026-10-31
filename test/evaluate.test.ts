import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { describe, it } from "node:test";
import type { Call } from "../src/arms.ts";
import { captureFromCsv } from "../src/capture.ts";
import { restTransport, type Metadata, type Transport } from "../src/cloudflare.ts";
import { build, readClefJsonl } from "../src/dataset.ts";
import { evaluate, parseArm, renderMarkdown } from "../src/evaluate.ts";
import { FAKE_NAMES, humanRule, QUESTIONS } from "../scripts/synthetic.ts";

async function splitFixture() {
  const records = captureFromCsv(await readFile("fixtures/leads.csv", "utf8"), QUESTIONS).records;
  const out = build(records, { heldOut: 60, template: "mistral", clefModel: "@cf/cloudflare/clef-flash", redact: { names: FAKE_NAMES } });
  return { train: readClefJsonl(out.files["clef/train.jsonl"]), heldOut: readClefJsonl(out.files["clef/heldout.jsonl"]) };
}

/** A fake transport that answers like each documented API, using a rule as the "model". */
function fakeTransport(seen: Array<{ call: Call; meta: Metadata }>): Transport {
  const answer = (state: unknown) => humanRule(state as Parameters<typeof humanRule>[0]);
  return {
    async send(call, meta) {
      seen.push({ call, meta });
      if (call.transport === "workers-ai" && "questions" in call.input) {
        const choice = answer(call.input.state);
        const probabilities = { accept: 0.1, skip: 0.1, needs_info: 0.1, [choice]: 0.8 };
        return { model: "clef-flash", answers: { decision: { type: "choice", choice, probabilities, confidence: 0.8 } }, usage: { input_tokens: 1, output_tokens: 0 } };
      }
      if (call.transport === "workers-ai") return { response: "I think we should accept" }; // prose: must count as a miss
      if (call.transport === "chat") throw new Error("429 rate limited");
      throw new Error("unexpected");
    },
  };
}

describe("evaluate", () => {
  it("scores every arm on the same records, counting failures and prose as misses", async () => {
    const { train, heldOut } = await splitFixture();
    const seen: Array<{ call: Call; meta: Metadata }> = [];
    const arms = [
      parseArm("logged=replay"),
      parseArm("majority=majority"),
      parseArm("clef=systemone:@cf/cloudflare/clef-flash"),
      parseArm("lora=lora:@cf/mistral/mistral-7b-instruct-v0.2-lora|leads-v1|mistral"),
      parseArm("frontier=chat:openai/gpt-5.5"),
    ];
    let t = 0;
    const rep = await evaluate(heldOut, arms, { transport: fakeTransport(seen), train, runId: "test-run", concurrency: 3, clock: () => (t += 5) });
    const by = Object.fromEntries(rep.arms.map((a) => [a.name, a]));
    assert.equal(rep.n, 60);
    for (const a of rep.arms) assert.equal(a.questions.decision.n, 60);
    // The fake Clef answers with the labeling rule, so it only misses the 4% label noise.
    assert.ok(by.clef.exactMatch > 0.85, `clef ${by.clef.exactMatch}`);
    assert.equal(by.lora.exactMatch, 0);
    assert.equal(by.lora.questions.decision.answered, 0);
    assert.equal(by.frontier.errors.length, 60);
    assert.match(by.frontier.errors[0].error, /429/);
    assert.equal(by.logged.latencyMs, null);
    assert.ok(by.clef.latencyMs!.p50! > 0);
    // Every call is tagged as eval traffic so capture skips it.
    assert.equal(seen.length, 180);
    assert.ok(seen.every((s) => s.meta.clef_loop_eval === true && s.meta.run === "test-run"));
    // Pairwise tests are against the first arm.
    assert.deepEqual(rep.pairwise.map((p) => p.b), ["majority", "clef", "lora", "frontier"]);
    const md = renderMarkdown(rep, "Synthetic data.");
    assert.match(md, /\| clef \| systemone \|/);
    assert.match(md, /not production latency/);
  });

  it("needs training records for the majority arm and a transport for model arms", async () => {
    const { heldOut } = await splitFixture();
    await assert.rejects(evaluate(heldOut, [parseArm("m=majority")], { runId: "r" }), /needs the training records/);
    await assert.rejects(evaluate(heldOut, [parseArm("c=chat:openai/gpt-5.5")], { runId: "r" }), /needs a transport/);
    await assert.rejects(evaluate(heldOut, [parseArm("a=replay"), parseArm("a=majority")], { runId: "r" }), /duplicate arm/);
  });
});

describe("evaluate over HTTP", () => {
  it("scores a self-hosted System One server (like SGLang's /v1/systemone) through the real fetch", async () => {
    const { createServer } = await import("node:http");
    const server = createServer((req, res) => {
      let body = "";
      req.on("data", (c) => (body += c));
      req.on("end", () => {
        const { state, questions } = JSON.parse(body);
        assert.ok(questions.decision && req.url === "/v1/systemone");
        const choice = humanRule(state);
        res.setHeader("Content-Type", "application/json");
        res.end(
          JSON.stringify({
            model: "Cloudflare/clef-flash",
            answers: { decision: { type: "choice", choice, probabilities: { accept: 0.05, skip: 0.05, needs_info: 0.05, [choice]: 0.9 }, confidence: 0.9 } },
            usage: { input_tokens: 10, output_tokens: 0 },
          }),
        );
      });
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    try {
      const { port } = server.address() as { port: number };
      const { heldOut } = await splitFixture();
      const rep = await evaluate(heldOut, [parseArm(`local=systemone-url:http://127.0.0.1:${port}/v1/systemone`)], {
        transport: restTransport({ gatewayId: "unused" }),
        runId: "http",
      });
      assert.equal(rep.arms[0].errors.length, 0);
      assert.equal(rep.arms[0].questions.decision.answered, 60);
    } finally {
      server.close();
    }
  });
});

describe("parseArm", () => {
  it("parses each kind", () => {
    assert.deepEqual(parseArm("x=systemone-url:http://localhost:30000/v1/systemone").spec, { kind: "systemone-url", url: "http://localhost:30000/v1/systemone" });
    assert.deepEqual(parseArm("x=lora:@cf/google/gemma-7b-it-lora||gemma").spec, { kind: "lora", model: "@cf/google/gemma-7b-it-lora", template: "gemma" });
    assert.throws(() => parseArm("x=chat"), /needs a model/);
    assert.throws(() => parseArm("nonsense"), /name=kind/);
    assert.throws(() => parseArm("x=lora:m|f|chatml"), /unknown template/);
  });
});

describe("restTransport", () => {
  function recorder(responses: Response[]) {
    const calls: Array<{ url: string; init: RequestInit }> = [];
    const f = (async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return responses.shift()!;
    }) as unknown as typeof fetch;
    return { f, calls };
  }

  it("calls Workers AI through the gateway with eval metadata, and unwraps the envelope", async () => {
    const { f, calls } = recorder([Response.json({ success: true, result: { answers: {} }, errors: [] })]);
    const t = restTransport({ accountId: "acct", apiToken: "tok", gatewayId: "gw", fetch: f });
    const out = await t.send({ transport: "workers-ai", model: "@cf/cloudflare/clef-flash", input: { model: "clef-flash" } }, { clef_loop_eval: true, arm: "clef" });
    assert.deepEqual(out, { answers: {} });
    assert.equal(calls[0].url, "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef-flash");
    const h = calls[0].init.headers as Record<string, string>;
    assert.equal(h.Authorization, "Bearer tok");
    assert.equal(h["cf-aig-gateway-id"], "gw");
    assert.deepEqual(JSON.parse(h["cf-aig-metadata"]), { clef_loop_eval: true, arm: "clef" });
  });

  it("sends chat models to the OpenAI-compatible endpoint", async () => {
    const { f, calls } = recorder([Response.json({ choices: [{ message: { content: "skip" } }] })]);
    const t = restTransport({ accountId: "acct", apiToken: "tok", gatewayId: "gw", fetch: f });
    await t.send({ transport: "chat", body: { model: "openai/gpt-5.5", messages: [] } }, {});
    assert.equal(calls[0].url, "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/chat/completions");
  });

  it("retries 429 and 5xx, but not 4xx", async () => {
    const sleeps: number[] = [];
    const sleep = async (ms: number) => void sleeps.push(ms);
    const a = recorder([new Response("busy", { status: 429 }), new Response("oops", { status: 503 }), Response.json({ success: true, result: 1 })]);
    assert.equal(await restTransport({ accountId: "a", apiToken: "t", gatewayId: "g", fetch: a.f, sleep }).send({ transport: "url", url: "http://x/v1/systemone", body: {} }, {}), 1);
    assert.deepEqual(sleeps, [250, 500]);
    const b = recorder([new Response("bad", { status: 400 })]);
    await assert.rejects(restTransport({ accountId: "a", apiToken: "t", gatewayId: "g", fetch: b.f, sleep }).send({ transport: "url", url: "http://x", body: {} }, {}), /400 bad/);
    assert.equal(b.calls.length, 1);
  });

  it("needs credentials only for Cloudflare calls", async () => {
    const t = restTransport({ gatewayId: "g", fetch: (async () => Response.json({ ok: 1 })) as unknown as typeof fetch });
    await assert.rejects(t.send({ transport: "chat", body: {} }, {}), /CLOUDFLARE_ACCOUNT_ID/);
    assert.deepEqual(await t.send({ transport: "url", url: "http://x", body: {} }, {}), { ok: 1 });
  });
});
