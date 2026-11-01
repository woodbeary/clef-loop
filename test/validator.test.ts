import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { describe, it } from "node:test";
import type { RouteStatus } from "../src/route-book.ts";
import { handle } from "../worker/handler.ts";
import { verdict } from "../validator/rules.ts";
import { createValidatorServer, type Rule } from "../validator/server.ts";
import { humanRule, QUESTIONS, syntheticLeads } from "../scripts/synthetic.ts";
import { harness, req, vars } from "./worker-harness.ts";

async function withValidator<T>(rule: Rule, f: (fetcher: { fetch(r: Request): Promise<Response> }, seen: unknown[]) => Promise<T>): Promise<T> {
  const seen: unknown[] = [];
  const server = createValidatorServer((state, questions) => {
    seen.push({ state, questions });
    return rule(state, questions);
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const { port } = server.address() as AddressInfo;
  const fetcher = {
    fetch: async (r: Request) => {
      const u = new URL(r.url);
      const body = r.method === "GET" || r.method === "HEAD" ? undefined : await r.text();
      return fetch(`http://127.0.0.1:${port}${u.pathname}`, { method: r.method, headers: r.headers, body });
    },
  };
  try {
    return await f(fetcher, seen);
  } finally {
    server.close();
  }
}

const leads = syntheticLeads().slice(0, 40);

describe("validator rule and server", () => {
  it("agrees with the rule that labeled the synthetic fixtures", () => {
    for (const r of syntheticLeads()) assert.deepEqual(verdict(r.state, QUESTIONS), { decision: humanRule(r.state as never) });
    assert.equal(verdict("free text", QUESTIONS), null);
  });

  it("answers /verdict and /health over HTTP, and rejects bad bodies", async () => {
    await withValidator(verdict as Rule, async (v) => {
      const ok = await v.fetch(new Request("https://validator/verdict", { method: "POST", body: JSON.stringify({ request_id: "r1", state: leads[0].state, questions: QUESTIONS }) }));
      assert.deepEqual(await ok.json(), { request_id: "r1", final: { decision: humanRule(leads[0].state as never) } });
      const unsure = await v.fetch(new Request("https://validator/verdict", { method: "POST", body: JSON.stringify({ request_id: "r2", state: "?", questions: QUESTIONS }) }));
      assert.equal(((await unsure.json()) as { final: unknown }).final, null);
      assert.equal((await v.fetch(new Request("https://validator/verdict", { method: "POST", body: "{nope" }))).status, 400);
      assert.equal((await v.fetch(new Request("https://validator/health"))).status, 200);
    });
  });
});

describe("POST /outcome", () => {
  it("turns a validator reject on a served request into a failure in that route's window, once", async () => {
    const h = harness("ok");
    const v = vars();
    const d = (await (await handle(req("decide", { key: "k1", state: { distance_miles: 3 }, questions: QUESTIONS }), v, h.deps)).json()) as { request_id: string; route: string };
    await h.settle();
    assert.equal(d.route, "tuned");
    const first = await handle(req("outcome", { request_id: d.request_id, ok: false, reason: "next action failed" }), v, h.deps);
    assert.equal(first.status, 200);
    assert.deepEqual(await first.json(), { counted: true, route: "tuned" });
    assert.deepEqual(await (await handle(req("outcome", { request_id: d.request_id, ok: false }), v, h.deps)).json(), { counted: false, route: "tuned" });
    const s = await h.status();
    assert.deepEqual([s.tuned.n, s.tuned.failures], [1, 1]);
    assert.equal((await handle(req("outcome", { request_id: "nope", ok: false }), v, h.deps)).status, 404);
    assert.equal((await handle(req("outcome", { request_id: d.request_id, ok: "no" }), v, h.deps)).status, 400);
  });

  it("counts a reject on a base-served request against base", async () => {
    const h = harness("ok");
    const v = vars({ MODE: "off" });
    const d = (await (await handle(req("decide", { key: "k2", state: { distance_miles: 3 }, questions: QUESTIONS }), v, h.deps)).json()) as { request_id: string };
    await h.settle();
    assert.deepEqual(await (await handle(req("outcome", { request_id: d.request_id, ok: false }), v, h.deps)).json(), { counted: true, route: "base" });
    assert.equal((await h.status()).base.failures, 1);
  });
});

describe("validator in the loop", () => {
  it("reverts the canary on well-formed wrong answers, and never shows the validator the answer", async () => {
    await withValidator(verdict as Rule, async (validator, seen) => {
      const h = harness("ok");
      h.deps.validator = validator;
      const v = vars({ CAPTURE_TENANTS: "acme" });
      let served = 0;
      for (const [i, lead] of leads.entries()) {
        const body = (await (await handle(req("decide", { key: `lead-${i}`, tenant: "acme", state: lead.state, questions: QUESTIONS }), v, h.deps)).json()) as { route: string };
        await h.settle();
        if (body.route === "tuned") served++;
        if ((await h.status()).reverted) break;
      }
      const s = await h.status();
      assert.equal(s.reverted, true);
      assert.match(s.reason!, /tuned requests failed/);
      assert.ok(s.tuned.failures > 0 && s.tuned.failures <= served);
      const after = (await (await handle(req("decide", { key: "after", state: leads[0].state, questions: QUESTIONS }), v, h.deps)).json()) as { route: string };
      assert.equal(after.route, "base");
      for (const call of seen) {
        const text = JSON.stringify(call);
        assert.ok(!/"decision":"(accept|skip|needs_info)"/.test(text), "the validator saw an answer");
        assert.ok(!text.includes("555-555"), "the validator saw unredacted PII");
      }
      const labels = (await h.labels("acme").list(null, 100)).labels;
      assert.ok(labels.length > 0);
      assert.ok(labels.every((l) => l.decided_by === "validator" && l.saw_suggestion === false));
    });
  });

  it("stores no validator label for a tenant without consent, and skips verdicts the check can't make", async () => {
    await withValidator(() => null, async (validator) => {
      const h = harness("ok");
      h.deps.validator = validator;
      const v = vars({ CAPTURE_TENANTS: "acme" });
      for (let i = 0; i < 6; i++) {
        await handle(req("decide", { key: `x${i}`, tenant: "acme", state: leads[i].state, questions: QUESTIONS }), v, h.deps);
        await h.settle();
      }
      const s = (await h.status()) as RouteStatus;
      assert.deepEqual([s.tuned.failures, s.reverted], [0, false]);
      assert.equal((await h.labels("acme").list(null, 10)).labels.length, 0);
    });
    await withValidator(verdict as Rule, async (validator) => {
      const h = harness("ok");
      h.deps.validator = validator;
      await handle(req("decide", { key: "y", tenant: "trial-user", state: leads[0].state, questions: QUESTIONS }), vars({ CAPTURE_TENANTS: "acme" }), h.deps);
      await h.settle();
      assert.equal((await h.labels("trial-user").list(null, 10)).labels.length, 0);
    });
  });

  it("scores the shadow arm against the validator without serving it", async () => {
    await withValidator(() => ({ decision: "skip" }), async (validator) => {
      const h = harness("ok");
      h.deps.validator = validator;
      for (let i = 0; i < 6; i++) {
        await handle(req("decide", { key: `s${i}`, state: leads[i].state, questions: QUESTIONS }), vars({ MODE: "shadow" }), h.deps);
        await h.settle();
      }
      const s = await h.status();
      assert.equal(s.tuned.failures, 6);
      assert.equal(s.base.failures, 0);
      assert.equal(s.reverted, true);
    });
  });
});
