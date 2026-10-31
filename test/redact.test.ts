import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { auditResidual, luhn, redactJson, redactText, sensitiveKeyLabel, type RedactionCounts } from "../src/redact.ts";

describe("redactText", () => {
  it("removes each kind of PII and counts it", () => {
    const counts: RedactionCounts = {};
    const out = redactText(
      "Hi, this is Dana Example. Email dana@example.com, call (555) 555-0123 or +1 555.555.0199. " +
        "Card 4111 1111 1111 1111, SSN 123-45-6789, from 192.168.0.12, at 742 Example St Apt 3. " +
        "Details: https://example.com/form?email=dana%40example.com",
      {},
      counts,
    );
    for (const leak of ["dana@", "555-0123", "555.0199", "4111", "123-45-6789", "192.168", "742 Example", "Dana Example", "?email="]) {
      assert.ok(!out.includes(leak), `leaked ${leak}: ${out}`);
    }
    assert.deepEqual(counts, { URL: 1, EMAIL: 1, CARD: 1, SSN: 1, PHONE: 2, IP: 1, ADDRESS: 1, NAME: 1 });
  });

  it("keeps numbers that aren't PII", () => {
    const text = "Distance 12.5 miles on 2026-10-31, 3 items, order 42, $1,250.00, ref 1234567890123456";
    // 1234567890123456 fails the Luhn check, so it isn't treated as a card. It is still a
    // long digit run, which the residual audit reports separately.
    assert.equal(redactText(text), text);
  });

  it("removes a name in a sign-off without a names list", () => {
    assert.equal(redactText("Mostly yard debris. Thanks, Morgan"), "Mostly yard debris. Thanks, [NAME]");
    assert.equal(redactText("Regards,\nAvery."), "Regards,\n[NAME].");
    assert.equal(redactText("thanks, everyone"), "thanks, everyone");
  });

  it("removes each capitalized part of a listed name", () => {
    assert.equal(redactText("Casey said the testcase failed. Casey Testcase", { names: ["Casey Testcase"] }), "[NAME] said the testcase failed. [NAME]");
  });

  it("removes listed names case-insensitively, whole words only", () => {
    const out = redactText("riley SAMPLE asked; Rileyson did not", { names: ["Riley Sample"] });
    assert.equal(out, "[NAME] asked; Rileyson did not");
  });
});

describe("redactJson", () => {
  it("replaces values under sensitive keys and walks nested data", () => {
    const counts: RedactionCounts = {};
    const out = redactJson(
      {
        customer_name: "Dana Example",
        contactEmail: "dana@example.com",
        phone: 5555550123,
        service: "garage cleanout",
        zip_code: "00000",
        photos_attached: true,
        history: [{ note: "texted 555-555-0100" }],
        empty_name: "",
      },
      {},
      counts,
    );
    assert.deepEqual(out, {
      customer_name: "[NAME]",
      contactEmail: "[EMAIL]",
      phone: "[PHONE]",
      service: "garage cleanout",
      zip_code: "00000",
      photos_attached: true,
      history: [{ note: "texted [PHONE]" }],
      empty_name: "",
    });
    assert.deepEqual(counts, { NAME: 1, EMAIL: 1, PHONE: 2 });
  });

  it("respects keepKeys", () => {
    assert.deepEqual(redactJson({ service_name: "yard debris" }, { keepKeys: ["service_name"] }), { service_name: "yard debris" });
  });

  it("labels keys by their tokens, not substrings", () => {
    assert.equal(sensitiveKeyLabel("billing_address"), "ADDRESS");
    assert.equal(sensitiveKeyLabel("firstName"), "NAME");
    assert.equal(sensitiveKeyLabel("zip"), undefined);
    assert.equal(sensitiveKeyLabel("hotel"), undefined);
    assert.equal(sensitiveKeyLabel("shipping_method"), undefined);
  });
});

describe("auditResidual", () => {
  it("flags what still looks like PII, with masked samples and paths", () => {
    const findings = auditResidual({ a: "reach me at someone at mail dot com or x@example.org", b: ["ref 1234567890123456"], c: "ping @handle_9" });
    assert.deepEqual(
      findings.map((f) => [f.path, f.kind]),
      [
        ["$.a", "email-like"],
        ["$.b[0]", "long-digit-run"],
        ["$.c", "at-handle"],
      ],
    );
    assert.ok(findings.every((f) => !f.sample.includes("1234567890")));
  });

  it("is clean on redacted output", () => {
    assert.deepEqual(auditResidual(redactJson({ m: "call 555-555-0100, mail a@example.com" })), []);
  });
});

describe("luhn", () => {
  it("accepts test card numbers and rejects others", () => {
    assert.equal(luhn("4111111111111111"), true);
    assert.equal(luhn("4111111111111112"), false);
    assert.equal(luhn("123"), false);
  });
});
