// Drives the canary Worker's request handler in Node with a fake AI binding and an
// in-memory route store that uses the same revert rule as the Durable Object.

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { parseRevertConfig } from "../src/canary.ts";
import { memoryStorage, RouteBook, type RouteStatus } from "../src/route-book.ts";
import { authorized, handle, type AiBinding, type Deps, type RouteStore, type Vars } from "../worker/handler.ts";
import { QUESTIONS } from "../scripts/synthetic.ts";

const TOKEN = "test-token-not-a-secret";
const LORA = { kind: "lora", model: "@cf/mistral/mistral-7b-instruct-v0.2-lora", lora: "leads-v1", template: "mistral" };

function vars(over: Partial<Vars> = {}): Vars {
  return {
    CLEF_LOOP_TOKEN: TOKEN,
    GATEWAY_ID: "clef-loop",
    BASE_ARM: JSON.stringify({ kind: "systemone", model: "@cf/cloudflare/clef-flash" }),
    TUNED_ARM: JSON.stringify(LORA),
    MODE: "canary",
    CANARY_PERCENT: "100",
    ROUTE_VERSION: "v1",
    TUNED_TIMEOUT_MS: "50",
    ...over,
  };
}

type TunedBehavior = "ok" | "error" | "prose" | "slow";

function harness(tuned: TunedBehavior = "ok") {
  const runs: Array<{ model: string; input: Record<string, unknown>; metadata: Record<string, unknown> }> = [];
  const patches: Array<{ id: string; data: unknown }> = [];
  let n = 0;
  const ai: AiBinding = {
    aiGatewayLogId: null,
    async run(model, input, options) {
      runs.push({ model, input: input as Record<string, unknown>, metadata: options?.gateway?.metadata ?? {} });
      ai.aiGatewayLogId = `log-${++n}`;
      if (model === "@cf/cloudflare/clef-flash") {
        return { model: "clef-flash", answers: { decision: { type: "choice", choice: "skip", probabilities: { accept: 0.1, skip: 0.8, needs_info: 0.1 }, confidence: 0.7 } }, usage: { input_tokens: 1, output_tokens: 0 } };
      }
      if (tuned === "error") throw new Error("5007: model unavailable");
      if (tuned === "prose") return { response: "Probably accept, it looks fine" };
      if (tuned === "slow") await new Promise((r) => setTimeout(r, 200));
      return { response: "accept" };
    },
    gateway: () => ({ async patchLog(id, data) { patches.push({ id, data }); } }),
  };
  const route: RouteStore = new RouteBook(memoryStorage(), parseRevertConfig({ REVERT_MIN_SAMPLES: "5", REVERT_WINDOW: "20" }));
  const pending: Promise<unknown>[] = [];
  const deps: Deps = { ai, route, waitUntil: (p) => void pending.push(p) };
  return { deps, runs, patches, settle: () => Promise.all(pending), status: () => route.status() as Promise<RouteStatus> };
}

function req(path: string, body?: unknown, token = TOKEN): Request {
  return new Request(`https://canary.example/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

const decideBody = { key: "lead-1", state: { service: "yard debris", distance_miles: 4 }, questions: QUESTIONS };

describe("canary Worker", () => {
  it("rejects requests without the token", async () => {
    const h = harness();
    assert.equal((await handle(req("decide", decideBody, "wrong"), vars(), h.deps)).status, 401);
    assert.equal((await handle(req("decide", decideBody), vars({ CLEF_LOOP_TOKEN: undefined }), h.deps)).status, 401);
    assert.equal(authorized(`Bearer ${TOKEN}`, TOKEN), true);
    assert.equal(authorized(`Bearer ${TOKEN}x`, TOKEN), false);
  });

  it("serves the tuned model in the canary bucket and tags the gateway log", async () => {
    const h = harness("ok");
    const res = await handle(req("decide", decideBody), vars(), h.deps);
    const body = (await res.json()) as Record<string, unknown>;
    await h.settle();
    assert.deepEqual(body.decision, { decision: "accept" });
    assert.equal(body.route, "tuned");
    assert.equal(body.fallback, null);
    assert.equal(body.log_id, "log-1");
    assert.equal(h.runs.length, 1);
    assert.equal(h.runs[0].input.lora, "leads-v1");
    assert.equal(h.runs[0].metadata.route, "tuned");
    assert.equal(h.runs[0].metadata.req, body.request_id);
    assert.equal((await h.status()).tuned.n, 1);
  });

  for (const behavior of ["error", "prose", "slow"] as const) {
    it(`falls back to base when the tuned model is ${behavior === "prose" ? "unparseable" : behavior}`, async () => {
      const h = harness(behavior);
      const body = (await (await handle(req("decide", decideBody), vars(), h.deps)).json()) as Record<string, unknown>;
      await h.settle();
      assert.equal(body.route, "base");
      assert.deepEqual(body.decision, { decision: "skip" });
      assert.match(String(body.fallback), behavior === "error" ? /5007/ : behavior === "prose" ? /not an option id/ : /timed out/);
      assert.equal(h.runs.at(-1)!.metadata.route, "fallback");
      const s = await h.status();
      assert.deepEqual([s.tuned.n, s.tuned.failures, s.base.n, s.base.failures], [1, 1, 1, 0]);
    });
  }

  it("reverts the whole route after repeated failures, then serves base without trying tuned", async () => {
    const h = harness("error");
    for (let i = 0; i < 5; i++) {
      await handle(req("decide", { ...decideBody, key: `k${i}` }), vars(), h.deps);
      await h.settle();
    }
    assert.equal((await h.status()).reverted, true);
    const before = h.runs.length;
    const body = (await (await handle(req("decide", decideBody), vars(), h.deps)).json()) as Record<string, unknown>;
    assert.equal(body.route, "base");
    assert.equal(body.fallback, null);
    assert.equal(h.runs.length, before + 1);
    assert.equal(h.runs.at(-1)!.model, "@cf/cloudflare/clef-flash");
    const status = (await (await handle(req("route"), vars(), h.deps)).json()) as RouteStatus;
    assert.equal(status.reverted, true);
    assert.match(status.reason!, /5 of the last 5 tuned requests failed/);
    await handle(req("route/reset", {}), vars(), h.deps);
    assert.equal((await h.status()).reverted, false);
  });

  it("serves base in shadow mode and runs tuned on the side", async () => {
    const h = harness("ok");
    const body = (await (await handle(req("decide", decideBody), vars({ MODE: "shadow" }), h.deps)).json()) as Record<string, unknown>;
    await h.settle();
    assert.equal(body.route, "base");
    assert.deepEqual(h.runs.map((r) => r.metadata.route), ["base", "shadow"]);
    assert.equal(body.log_id, "log-1");
    const s = await h.status();
    assert.deepEqual([s.tuned.n, s.tuned.failures, s.base.n], [1, 0, 1]);
  });

  it("serves base only when MODE is off or no tuned arm is set", async () => {
    for (const v of [vars({ MODE: "off" }), vars({ TUNED_ARM: "" })]) {
      const h = harness("ok");
      const body = (await (await handle(req("decide", decideBody), v, h.deps)).json()) as Record<string, unknown>;
      assert.equal(body.route, "base");
      assert.equal(h.runs.length, 1);
    }
  });

  it("attaches the human's final decision to the log", async () => {
    const h = harness();
    const res = await handle(req("feedback", { log_id: "log-9", final: { decision: "accept" }, route: "tuned" }), vars(), h.deps);
    assert.equal(res.status, 204);
    assert.deepEqual(h.patches, [{ id: "log-9", data: { metadata: { final: '{"decision":"accept"}', route: "tuned", route_version: "v1" } } }]);
  });

  it("rejects malformed requests with 400", async () => {
    const h = harness();
    const bad = [
      req("decide", { state: "x", questions: {} }),
      req("decide", { questions: QUESTIONS }),
      req("feedback", { final: { decision: "accept" } }),
      req("feedback", { log_id: "x", final: { decision: 1 } }),
      new Request("https://canary.example/decide", { method: "POST", headers: { Authorization: `Bearer ${TOKEN}` }, body: "{not json" }),
    ];
    for (const r of bad) assert.equal((await handle(r, vars(), h.deps)).status, 400);
    assert.equal((await handle(req("nope"), vars(), h.deps)).status, 404);
  });
});
