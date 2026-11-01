// Drives the canary Worker's request handler in Node with a fake AI binding and the
// same RouteBook and LabelBook code the Durable Objects run, over in-memory storage.

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type { Label } from "../src/labels.ts";
import type { RouteStatus } from "../src/route-book.ts";
import { authorized, handle, stateHash, type AiBinding } from "../worker/handler.ts";
import { QUESTIONS } from "../scripts/synthetic.ts";
import { decideBody, harness, req, TOKEN, vars } from "./worker-harness.ts";

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
    assert.equal(body.log_id, undefined);
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
    assert.equal(h.runs[1].metadata.req, body.request_id);
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

  describe("labels", () => {
    const v = vars({ MODE: "off", CAPTURE_TENANTS: "acme" });
    const decided_at = "2026-10-30T12:00:00.000Z";
    const label = (over: Record<string, unknown> = {}) => ({ request_id: "req-1", tenant: "acme", final: { decision: "accept" }, decided_by: "human", decided_at, saw_suggestion: true, ...over });

    it("stores a human label by request id, with who decided, when, and whether they saw the suggestion", async () => {
      const h = harness();
      const decided = (await (await handle(req("decide", { ...decideBody, tenant: "acme" }), v, h.deps)).json()) as Record<string, unknown>;
      const res = await handle(req("feedback", label({ request_id: decided.request_id })), v, h.deps);
      assert.equal(res.status, 204);
      const listed = (await (await handle(req("labels?tenant=acme"), v, h.deps)).json()) as { labels: Label[]; next: string | null };
      assert.equal(listed.labels.length, 1);
      const l = listed.labels[0];
      assert.deepEqual(
        { request_id: l.request_id, final: l.final, decided_by: l.decided_by, decided_at: l.decided_at, saw_suggestion: l.saw_suggestion },
        { request_id: decided.request_id, final: { decision: "accept" }, decided_by: "human", decided_at, saw_suggestion: true },
      );
      assert.equal(h.runs[0].metadata.req, decided.request_id);
      assert.equal(listed.next, null);
    });

    it("refuses agent-made labels", async () => {
      const h = harness();
      const res = await handle(req("feedback", label({ decided_by: "agent" })), v, h.deps);
      assert.equal(res.status, 422);
      assert.match(((await res.json()) as { error: string }).error, /agent labels are refused/);
      assert.equal((await h.labels("acme").list(null, 10)).labels.length, 0);
    });

    it("requires decided_by, decided_at, and for a person, whether they saw the suggestion", async () => {
      const h = harness();
      const bad = [
        label({ decided_by: undefined }),
        label({ decided_by: "model" }),
        label({ decided_at: undefined }),
        label({ decided_at: "yesterday" }),
        label({ decided_at: "2099-01-01T00:00:00Z" }),
        label({ saw_suggestion: undefined }),
        label({ decided_by: "validator", saw_suggestion: true }),
        label({ request_id: undefined }),
        label({ final: { decision: 1 } }),
      ];
      for (const b of bad) assert.equal((await handle(req("feedback", b), v, h.deps)).status, 400, JSON.stringify(b));
      assert.equal((await handle(req("feedback", label({ decided_by: "validator", saw_suggestion: undefined })), v, h.deps)).status, 204);
    });

    it("refuses labels for a tenant that didn't opt in, since nothing was logged", async () => {
      const h = harness();
      assert.equal((await handle(req("feedback", label({ tenant: "trial-user" })), v, h.deps)).status, 409);
    });

    it("keeps a human label over a later validator label, and lets a person correct it", async () => {
      const h = harness();
      assert.equal((await handle(req("feedback", label()), v, h.deps)).status, 204);
      assert.equal((await handle(req("feedback", label({ decided_by: "validator", saw_suggestion: false, final: { decision: "skip" } })), v, h.deps)).status, 409);
      assert.equal((await handle(req("feedback", label({ final: { decision: "needs_info" } })), v, h.deps)).status, 204);
      const all = await h.labels("acme").list(null, 10);
      assert.deepEqual(all.labels.map((l) => l.final.decision), ["needs_info"]);
    });

    it("pages through labels and deletes a tenant's labels when consent is withdrawn", async () => {
      const h = harness();
      for (const id of ["a", "b", "c"]) await handle(req("feedback", label({ request_id: id })), v, h.deps);
      const p1 = (await (await handle(req("labels?tenant=acme&limit=2"), v, h.deps)).json()) as { labels: Label[]; next: string };
      assert.equal(p1.labels.length, 2);
      const p2 = (await (await handle(req(`labels?tenant=acme&after=${encodeURIComponent(p1.next)}`), v, h.deps)).json()) as { labels: Label[]; next: string | null };
      assert.deepEqual([...p1.labels, ...p2.labels].map((l) => l.request_id).sort(), ["a", "b", "c"]);
      assert.equal(p2.next, null);
      const del = await handle(new Request("https://canary.example/labels?tenant=acme", { method: "DELETE", headers: { Authorization: `Bearer ${TOKEN}` } }), v, h.deps);
      assert.equal(del.status, 200);
      assert.equal((await h.labels("acme").list(null, 10)).labels.length, 0);
    });

    it("returns each caller its own request id when calls finish out of order", async () => {
      const h = harness();
      const order: Array<() => void> = [];
      const ai = h.deps.ai;
      const slowAi: AiBinding = {
        async run(model, input, options) {
          await new Promise<void>((r) => order.push(r));
          return ai.run(model, input, options);
        },
      };
      const deps = { ...h.deps, ai: slowAi };
      const a = handle(req("decide", { ...decideBody, key: "a", tenant: "acme" }), v, deps);
      const b = handle(req("decide", { ...decideBody, key: "b", tenant: "acme" }), v, deps);
      while (order.length < 2) await new Promise((r) => setTimeout(r, 1));
      order[1]();
      const rb = (await (await b).json()) as Record<string, unknown>;
      order[0]();
      const ra = (await (await a).json()) as Record<string, unknown>;
      assert.notEqual(ra.request_id, rb.request_id);
      assert.equal(h.runs[0].metadata.req, rb.request_id);
      assert.equal(h.runs[1].metadata.req, ra.request_id);
    });
  });

  it("rejects malformed requests with 400", async () => {
    const h = harness();
    const bad = [
      req("decide", { state: "x", questions: {} }),
      req("decide", { questions: QUESTIONS }),
      req("feedback", { final: { decision: "accept" } }),
      req("feedback", { request_id: "x", tenant: "acme", final: { decision: 1 } }),
      req("labels"),
      new Request("https://canary.example/decide", { method: "POST", headers: { Authorization: `Bearer ${TOKEN}` }, body: "{not json" }),
    ];
    for (const r of bad) assert.equal((await handle(r, vars(), h.deps)).status, 400);
    assert.equal((await handle(req("nope"), vars(), h.deps)).status, 404);
  });

  describe("capture shim", () => {
    const pii = {
      key: "lead-7",
      state: {
        customer_name: "Dana Example",
        phone: "555-555-0142",
        message: "Hi, this is Dana Example. Call 555-555-0142 or dana@example.com about the garage cleanout.",
        distance_miles: 6,
      },
      questions: QUESTIONS,
    };

    it("redacts the state before the model sees it and logs nothing for a tenant without consent", async () => {
      const h = harness();
      const body = (await (await handle(req("decide", { ...pii, tenant: "trial-user" }), vars({ MODE: "off", CAPTURE_TENANTS: "acme" }), h.deps)).json()) as Record<string, unknown>;
      assert.equal(body.captured, false);
      assert.equal(h.runs.length, 1);
      assert.equal(h.runs[0].collectLog, false);
      const sent = JSON.stringify(h.runs[0].input);
      for (const leak of ["Dana", "555-555-0142", "dana@example.com"]) assert.ok(!sent.includes(leak), `${leak} reached the model`);
      assert.match(sent, /\[NAME\]/);
      assert.match(sent, /\[PHONE\]/);
      assert.match(sent, /\[EMAIL\]/);
      assert.equal(h.events[0].capture_skipped, "no consent");
    });

    it("turns gateway logging on per request only for a tenant that opted in", async () => {
      const h = harness();
      const v = vars({ MODE: "off", CAPTURE_TENANTS: "acme, other" });
      const body = (await (await handle(req("decide", { ...pii, tenant: "acme" }), v, h.deps)).json()) as Record<string, unknown>;
      assert.equal(body.captured, true);
      assert.equal(h.runs[0].collectLog, true);
      assert.equal(h.runs[0].metadata.tenant, "acme");
      assert.ok(!JSON.stringify(h.runs[0].input).includes("Dana"));
      await handle(req("decide", { ...pii }), v, h.deps);
      assert.equal(h.runs[1].collectLog, false);
      assert.equal(h.runs[1].metadata.tenant, "default");
    });

    it("skips capture for an opted-in tenant when the residual audit still finds PII", async () => {
      const h = harness();
      const state = { message: "acct 9876 5432 1098, ping @dana_example", distance_miles: 3 };
      const body = (await (await handle(req("decide", { key: "k", tenant: "acme", state, questions: QUESTIONS }), vars({ MODE: "off", CAPTURE_TENANTS: "acme" }), h.deps)).json()) as Record<string, unknown>;
      assert.equal(body.captured, false);
      assert.equal(h.runs[0].collectLog, false);
      assert.equal(h.events[0].capture_skipped, "residual PII");
    });

    it("keeps a metadata-only event for every decision: pTop, margin, latency and a state hash, never the state", async () => {
      const h = harness();
      const body = (await (await handle(req("decide", { ...pii, tenant: "trial-user" }), vars({ MODE: "off" }), h.deps)).json()) as Record<string, unknown>;
      assert.equal(h.events.length, 1);
      const e = h.events[0];
      assert.equal(e.req, body.request_id);
      assert.equal(e.route, "base");
      assert.equal(e.answers.decision.p_top, 0.8);
      assert.ok(Math.abs(e.answers.decision.margin - 0.7) < 1e-9);
      assert.equal(e.latency_ms, 7);
      assert.match(e.state_hash, /^[0-9a-f]{16}$/);
      assert.equal(e.state_hash, await stateHash(h.runs[0].input.state as never));
      assert.ok(e.redactions >= 4);
      const line = JSON.stringify(e);
      for (const leak of ["Dana", "555-555", "example.com", "garage"]) assert.ok(!line.includes(leak), `${leak} in the event`);
      assert.deepEqual(body.answers, { decision: { p_top: 0.8, margin: e.answers.decision.margin } });
    });

    it("flags low-confidence answers for a person when ABSTAIN_BELOW is set", async () => {
      const h = harness();
      const low = (await (await handle(req("decide", decideBody), vars({ MODE: "off", ABSTAIN_BELOW: "0.9" }), h.deps)).json()) as Record<string, unknown>;
      assert.deepEqual(low.needs_person, ["decision"]);
      const ok = (await (await handle(req("decide", decideBody), vars({ MODE: "off", ABSTAIN_BELOW: "0.75" }), h.deps)).json()) as Record<string, unknown>;
      assert.deepEqual(ok.needs_person, []);
      assert.deepEqual(h.events.map((e) => e.needs_person), [["decision"], []]);
    });

    it("keeps keys the operator marks as safe, and rejects a malformed tenant", async () => {
      const h = harness();
      await handle(req("decide", { key: "k", state: { service_name: "junk hauling", distance_miles: 2 }, questions: QUESTIONS }), vars({ MODE: "off", REDACT_KEEP_KEYS: "service_name" }), h.deps);
      assert.equal((h.runs[0].input.state as Record<string, unknown>).service_name, "junk hauling");
      assert.equal((await handle(req("decide", { ...pii, tenant: "bad tenant!" }), vars(), h.deps)).status, 400);
    });
  });
});
