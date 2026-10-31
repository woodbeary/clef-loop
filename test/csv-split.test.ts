import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { parseCsv, toCsv } from "../src/csv.ts";
import { canonical, dedupe, split } from "../src/split.ts";
import type { DecisionRecord } from "../src/types.ts";
import { QUESTIONS } from "../scripts/synthetic.ts";

describe("csv", () => {
  it("handles quotes, commas, doubled quotes and newlines in fields", () => {
    const text = 'id,text\r\n1,"a, ""quoted""\nline"\n2,plain\n';
    assert.deepEqual(parseCsv(text), [
      { id: "1", text: 'a, "quoted"\nline' },
      { id: "2", text: "plain" },
    ]);
  });

  it("round-trips", () => {
    const rows = [["x,y", 'say "hi"'], ["multi\nline", ""]];
    const back = parseCsv(toCsv(["a", "b"], rows));
    assert.deepEqual(back, [
      { a: "x,y", b: 'say "hi"' },
      { a: "multi\nline", b: "" },
    ]);
  });

  it("rejects ragged rows and unterminated quotes", () => {
    assert.throws(() => parseCsv("a,b\n1\n"), /row 2 has 1 fields/);
    assert.throws(() => parseCsv('a\n"open\n'), /quoted field/);
  });
});

function rec(id: string, at: string, state: string, label: string): DecisionRecord {
  return { id, createdAt: at, state, questions: QUESTIONS, final: { decision: label }, source: "csv" };
}

describe("split", () => {
  const records = [
    rec("c", "2026-09-03T00:00:00Z", "three", "skip"),
    rec("a", "2026-09-01T00:00:00Z", "one", "accept"),
    rec("b", "2026-09-02T00:00:00Z", "two", "accept"),
    rec("d", "2026-09-04T00:00:00Z", "four", "needs_info"),
    rec("e", "2026-09-05T00:00:00Z", "five", "accept"),
    rec("a2", "2026-09-06T00:00:00Z", "one", "skip"), // same input as "a", labeled differently
  ];

  it("holds out the newest records and never puts the same input on both sides", () => {
    const s = split(records, { heldOut: 2 });
    assert.deepEqual(s.train.map((r) => r.id), ["a", "b", "c"]);
    assert.deepEqual(s.heldOut.map((r) => r.id), ["d", "e"]);
    assert.equal(s.cutoff, "2026-09-04T00:00:00Z");
    assert.equal(s.duplicatesRemoved, 1);
    assert.deepEqual(s.conflicts, [{ ids: ["a", "a2"] }]);
    const trainInputs = new Set(s.train.map((r) => canonical(r.state)));
    assert.ok(s.heldOut.every((r) => !trainInputs.has(canonical(r.state))));
  });

  it("accepts a fraction and rejects sizes that leave a side empty", () => {
    assert.equal(split(records, { heldOut: 0.4 }).heldOut.length, 2);
    assert.throws(() => split(records, { heldOut: 5 }), /between 1 and 4/);
    assert.throws(() => split(records, { heldOut: 0 }), /count or a fraction/);
    assert.throws(() => split(records, { heldOut: Number("abc") }), /count or a fraction/);
    assert.throws(() => split(records, { heldOut: 0.01 }), /between 1 and 4/);
  });

  it("dedupes regardless of key order", () => {
    const x = { ...rec("x", "2026-09-01T00:00:00Z", "", "accept"), state: { b: 1, a: 2 } };
    const y = { ...rec("y", "2026-09-02T00:00:00Z", "", "accept"), state: { a: 2, b: 1 } };
    assert.equal(dedupe([x, y]).unique.length, 1);
  });
});
