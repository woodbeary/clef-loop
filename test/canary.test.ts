import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { bucket, chooseRoute, EMPTY_STATE, parseRevertConfig, recordOutcome, type RevertState } from "../src/canary.ts";

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

describe("recordOutcome", () => {
  const cfg = { window: 200, minSamples: 50, maxFailureRate: 0.01 };
  const now = new Date("2026-11-01T00:00:00Z");
  const run = (outcomes: boolean[], s: RevertState = EMPTY_STATE) => outcomes.reduce((st, f) => recordOutcome(st, f, cfg, now), s);

  it("waits for the minimum sample before judging", () => {
    const s = run([true, true, true]);
    assert.equal(s.reverted, false);
    assert.equal(s.recent.length, 3);
  });

  it("reverts when failures pass the rate, and stays reverted", () => {
    const s = run([...Array(48).fill(false), true, false]); // 1 of 50 = 2% > 1%
    assert.equal(s.reverted, true);
    assert.match(s.reason!, /1 of the last 50 tuned requests failed \(2\.0% > 1\.0%\)/);
    assert.equal(s.revertedAt, now.toISOString());
    assert.equal(run(Array(300).fill(false), s).reverted, true);
  });

  it("holds at exactly the limit and slides the window", () => {
    const at = run([...Array(199).fill(false), true, true]); // window keeps the last 200: 2 failures = 1%
    assert.equal(at.recent.length, 200);
    assert.equal(at.reverted, false);
    const over = recordOutcome(at, true, cfg, now);
    assert.equal(over.reverted, true);
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
