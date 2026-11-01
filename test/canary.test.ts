import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { bucket, chooseRoute, judge, parseRevertConfig } from "../src/canary.ts";
import { memoryStorage, RouteBook } from "../src/route-book.ts";

describe("bucket and chooseRoute", () => {
  it("is sticky per key and re-buckets when the route version changes", () => {
    assert.equal(bucket("customer-42", "v1"), bucket("customer-42", "v1"));
    const moved = Array.from({ length: 200 }, (_, i) => bucket(`k${i}`, "v1") !== bucket(`k${i}`, "v2")).filter(Boolean).length;
    assert.ok(moved > 150);
  });

  it("sends close to the configured share to the tuned model", () => {
    const n = 20_000;
    const tuned = Array.from({ length: n }, (_, i) =>
      chooseRoute({ mode: "canary", percent: 10, key: `lead-${i}`, salt: "v1", tunedConfigured: true, reverted: false }),
    ).filter((r) => r === "tuned").length;
    assert.ok(Math.abs(tuned / n - 0.1) < 0.01, `got ${tuned / n}`);
  });

  it("serves base when off, shadowing, unconfigured or reverted", () => {
    const base = { percent: 100, key: "k", salt: "v1", tunedConfigured: true, reverted: false };
    assert.equal(chooseRoute({ ...base, mode: "canary" }), "tuned");
    assert.equal(chooseRoute({ ...base, mode: "off" }), "base");
    assert.equal(chooseRoute({ ...base, mode: "shadow" }), "base");
    assert.equal(chooseRoute({ ...base, mode: "canary", tunedConfigured: false }), "base");
    assert.equal(chooseRoute({ ...base, mode: "canary", reverted: true }), "base");
    assert.equal(chooseRoute({ ...base, mode: "canary", percent: 0 }), "base");
  });
});

describe("judge: revert on a Wilson lower bound, against base's own rate", () => {
  const cfg = { window: 200, minSamples: 50, maxFailureRate: 0.01 };
  const fails = (ok: number, failed: number) => [...Array(ok).fill(false), ...Array(failed).fill(true)];

  it("waits for the minimum sample before judging", () => {
    assert.equal(judge([true, true, true], [], cfg).revert, false);
  });

  it("does not revert on one transient failure between tuned requests 50 and 99", () => {
    for (const n of [50, 75, 99]) {
      const j = judge(fails(n - 1, 1), [], cfg);
      assert.equal(j.revert, false, `reverted at ${n}`);
      assert.ok(j.tuned.rate! > cfg.maxFailureRate);
    }
  });

  it("takes 2 failures in 50 tuned requests, 3 in 100 and 5 in 200, as docs/MEASURING.md says", () => {
    for (const [n, needed] of [[50, 2], [100, 3], [200, 5]]) {
      assert.equal(judge(fails(n - needed, needed), [], cfg).revert, true, `${needed} of ${n}`);
      assert.equal(judge(fails(n - needed + 1, needed - 1), [], cfg).revert, false, `${needed - 1} of ${n}`);
    }
  });

  it("reverts once the 95% lower bound passes the limit", () => {
    const j = judge(fails(47, 3), [], cfg);
    assert.equal(j.revert, true);
    assert.match(j.reason!, /3 of the last 50 tuned requests failed \(6\.0%, 95% lower bound 2\.1%\), above the limit 1\.0% \(base has 0 of 50 samples\)/);
  });

  it("holds tuned to base's own failure rate when base fails more than the limit", () => {
    const tuned = fails(90, 10);
    assert.equal(judge(tuned, fails(90, 10), cfg).revert, false);
    assert.equal(judge(tuned, fails(90, 10), cfg).limit, 0.1);
    const j = judge(tuned, fails(100, 0), cfg);
    assert.equal(j.revert, true);
    assert.match(j.reason!, /base 0\.0% of 100/);
    assert.equal(judge(tuned, fails(10, 10), cfg).limit, 0.01);
  });
});

describe("RouteBook", () => {
  const cfg = { window: 20, minSamples: 5, maxFailureRate: 0.01 };
  const now = () => new Date("2026-11-01T00:00:00Z");

  it("turns a validator reject into a failure once, so well-formed wrong answers revert the route", async () => {
    const book = new RouteBook(memoryStorage(), { ...cfg, maxFailureRate: 0.2 }, now);
    for (let i = 0; i < 5; i++) await book.record("base", false, `b${i}`);
    for (let i = 0; i < 5; i++) await book.record("tuned", false, `t${i}`);
    assert.equal((await book.status()).tuned.failures, 0);
    assert.deepEqual(await book.verdict("t0", false), { counted: true, route: "tuned" });
    assert.deepEqual(await book.verdict("t0", false), { counted: false, route: "tuned" });
    assert.equal((await book.status()).tuned.failures, 1);
    assert.equal(await book.admit(), true);
    await book.verdict("t1", false);
    assert.equal(await book.admit(), true);
    await book.verdict("t2", false);
    const s = await book.status();
    assert.equal(s.reverted, true);
    assert.equal(s.revertedAt, "2026-11-01T00:00:00.000Z");
    assert.match(s.reason!, /3 of the last 5 tuned requests failed \(60\.0%, 95% lower bound 23\.1%\)/);
    assert.equal(await book.admit(), false);
  });

  it("counts a verdict on a base request against base, and ignores unknown or expired requests", async () => {
    const book = new RouteBook(memoryStorage(), cfg, now);
    await book.record("base", false, "req-1");
    assert.deepEqual(await book.verdict("req-1", false), { counted: true, route: "base" });
    assert.equal((await book.status()).base.failures, 1);
    assert.equal((await book.verdict("nope", false)).counted, false);
    for (let i = 0; i < 20; i++) await book.record("base", false, `later-${i}`);
    assert.equal((await book.verdict("req-1", false)).counted, false);
    assert.equal((await book.status()).base.n, 20);
  });

  it("keeps storage bounded by the window and stays reverted until reset", async () => {
    const storage = memoryStorage();
    const book = new RouteBook(storage, cfg, now);
    for (let i = 0; i < 100; i++) await book.record("tuned", true, `k${i}`);
    assert.ok(storage.size() <= 2 * cfg.window + 3, `size ${storage.size()}`);
    assert.equal((await book.status()).reverted, true);
    for (let i = 0; i < 50; i++) await book.record("tuned", false);
    assert.equal((await book.status()).reverted, true);
    await book.reset();
    const s = await book.status();
    assert.equal(s.reverted, false);
    assert.equal(s.tuned.n, 0);
  });

  it("serializes concurrent updates", async () => {
    const book = new RouteBook(memoryStorage(), { ...cfg, window: 200 }, now);
    await Promise.all(Array.from({ length: 50 }, (_, i) => book.record("tuned", false, `c${i}`)));
    assert.equal((await book.status()).tuned.n, 50);
  });
});

describe("parseRevertConfig", () => {
  it("uses the documented defaults and validates overrides", () => {
    assert.deepEqual(parseRevertConfig({}), { window: 200, minSamples: 50, maxFailureRate: 0.01 });
    assert.throws(() => parseRevertConfig({ REVERT_WINDOW: "0" }), /positive integer/);
    assert.throws(() => parseRevertConfig({ REVERT_MAX_FAILURE_RATE: "2" }), /\[0, 1\)/);
    assert.throws(() => parseRevertConfig({ REVERT_WINDOW: "10", REVERT_MIN_SAMPLES: "20" }), /must not exceed/);
  });
});
