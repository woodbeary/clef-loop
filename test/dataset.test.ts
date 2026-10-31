import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { describe, it } from "node:test";
import { captureFromCsv } from "../src/capture.ts";
import { parseCsv } from "../src/csv.ts";
import { build, readClefJsonl } from "../src/dataset.ts";
import { validateQuestions } from "../src/schema.ts";
import { FAKE_NAMES, QUESTIONS } from "../scripts/synthetic.ts";

async function fixtureRecords() {
  return captureFromCsv(await readFile("fixtures/leads.csv", "utf8"), QUESTIONS).records;
}

describe("build", () => {
  it("writes Clef and LoRA datasets with no PII left in any file", async () => {
    const out = build(await fixtureRecords(), {
      heldOut: 60,
      template: "mistral",
      clefModel: "@cf/cloudflare/clef-flash",
      redact: { names: FAKE_NAMES },
    });
    assert.deepEqual(Object.keys(out.files).sort(), ["clef/heldout.jsonl", "clef/train.jsonl", "lora/heldout.jsonl", "lora/train.csv", "manifest.json"]);
    const all = Object.values(out.files).join("\n");
    assert.ok(!/@example\.com/.test(all), "email leaked");
    assert.ok(!/555-01\d\d/.test(all), "phone leaked");
    assert.ok(!/4111 1111/.test(all), "card leaked");
    assert.ok(!/\d+ Example St/.test(all), "address leaked");
    for (const n of FAKE_NAMES) assert.ok(!all.includes(n), `name leaked: ${n}`);
    for (const first of FAKE_NAMES.map((n) => n.split(" ")[0])) assert.ok(!all.includes(`Thanks, ${first}`), `first name leaked: ${first}`);
  });

  it("writes requests that match the documented Clef input schema, labels kept apart", async () => {
    const out = build(await fixtureRecords(), { heldOut: 60, template: "mistral", clefModel: "@cf/cloudflare/clef-flash", redact: { names: FAKE_NAMES } });
    const lines = out.files["clef/train.jsonl"].trim().split("\n").map((l) => JSON.parse(l));
    assert.equal(lines.length, 240);
    for (const l of lines) {
      assert.deepEqual(Object.keys(l.request).sort(), ["model", "questions", "state"]);
      assert.equal(l.request.model, "clef-flash");
      assert.deepEqual(validateQuestions(l.request.questions), []);
      assert.ok(["accept", "skip", "needs_info"].includes(l.label.decision));
    }
    const back = readClefJsonl(out.files["clef/heldout.jsonl"]);
    assert.equal(back.length, 60);
    assert.ok(back.every((r) => r.createdAt >= out.split.cutoff!));
  });

  it("writes the LoRA training CSV in the AutoTrain format: one `text` column in the chat template", async () => {
    const out = build(await fixtureRecords(), { heldOut: 60, template: "mistral", clefModel: "@cf/cloudflare/clef-flash", redact: { names: FAKE_NAMES } });
    const rows = parseCsv(out.files["lora/train.csv"]);
    assert.equal(rows.length, 240);
    assert.deepEqual(Object.keys(rows[0]), ["text"]);
    for (const r of rows) assert.match(r.text, /^<s>\[INST\] [\s\S]+ \[\/INST\] (accept|skip|needs_info)<\/s>$/);
  });

  it("records counts that add up in the manifest", async () => {
    const records = await fixtureRecords();
    const out = build(records, { heldOut: 0.2, template: "gemma", clefModel: "@cf/cloudflare/clef", redact: { names: FAKE_NAMES } });
    const m = JSON.parse(out.files["manifest.json"]);
    assert.equal(m.counts.input, records.length);
    assert.equal(m.counts.duplicates_removed, 3);
    assert.equal(m.counts.train + m.counts.heldout + m.counts.duplicates_removed, records.length);
    const sum = (o: Record<string, number>) => Object.values(o).reduce((a, b) => a + b, 0);
    assert.equal(sum(m.labels.train.decision), m.counts.train);
    assert.ok(m.redactions.PHONE > 0 && m.redactions.NAME > 0);
    assert.equal(m.residual_pii_findings, 0);
    assert.ok(m.time.train.to <= m.time.heldout.from);
    assert.match(out.files["lora/train.csv"], /<start_of_turn>model\n(accept|skip|needs_info)<end_of_turn>/);
  });

  it("refuses to write when the audit still finds PII, unless told to", async () => {
    const records = (await fixtureRecords()).slice(0, 20);
    records[3] = { ...records[3], state: { note: "reach me: someone(at)mail.com or acct 9876 5432 1098" } };
    assert.throws(
      () => build(records, { heldOut: 5, template: "mistral", clefModel: "@cf/cloudflare/clef-flash", redact: {} }),
      /PII audit found/,
    );
    const out = build(records, { heldOut: 5, template: "mistral", clefModel: "@cf/cloudflare/clef-flash", redact: {}, allowResidual: true });
    assert.ok(out.residual.length > 0);
  });
});
