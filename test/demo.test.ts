import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { runRevertDemo, TUNED_MEANS } from "../scripts/demo-revert.ts";

describe("demo: the canary reverts on validator rejects", () => {
  it("reverts on well-formed wrong answers alone, then serves base for a tuned-bucket key", async () => {
    const { lines, status, tunedServed, afterRoute } = await runRevertDemo(150);
    assert.equal(status.reverted, true);
    assert.match(status.reason!, /of the last 50 tuned requests failed .* above the limit 1\.0% \(base 0\.0% of \d+\)/);
    assert.equal(status.base.failures, 0);
    assert.ok(status.tuned.failures >= 20 && status.tuned.failures <= tunedServed);
    assert.equal(afterRoute, "base");
    assert.ok(lines.some((l) => l.includes(TUNED_MEANS)));
    assert.ok(lines.some((l) => l.includes("well-formed")));
  });
});
