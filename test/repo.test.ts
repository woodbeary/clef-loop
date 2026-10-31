// Repository checks: fixtures are reproducible, and nothing private is committed.

import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, it } from "node:test";
import { fixtureFiles } from "../scripts/make-fixtures.ts";

const SKIP_DIRS = new Set([".git", "node_modules", "dist", "out", "data", ".wrangler"]);

async function walk(dir: string): Promise<string[]> {
  const out: string[] = [];
  for (const e of await readdir(dir, { withFileTypes: true })) {
    if (e.isDirectory()) {
      if (!SKIP_DIRS.has(e.name)) out.push(...(await walk(join(dir, e.name))));
    } else if (e.name !== "package-lock.json") out.push(join(dir, e.name));
  }
  return out;
}

describe("fixtures", () => {
  it("match what scripts/make-fixtures.ts generates", async () => {
    for (const [path, text] of Object.entries(fixtureFiles())) {
      assert.equal(await readFile(path, "utf8"), text, `${path} is stale; run npm run fixtures`);
    }
  });
});

describe("public hygiene", () => {
  // Built from parts so this file doesn't match its own patterns.
  const at = "@";
  const checks: Array<[string, RegExp, (m: string) => boolean]> = [
    // Cloudflare account, zone and database ids are 32 hex characters.
    ["32-hex id", /\b[0-9a-f]{32}\b/g, () => true],
    ["email outside example.com", new RegExp(`[A-Za-z0-9._%+-]+${at}[A-Za-z0-9.-]+\\.[A-Za-z]{2,}`, "g"), (m) => !/@example\.(com|org|net)$/.test(m)],
    ["phone outside 555", /(?<![\d.])\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)/g, (m) => !/^\(?555\)?[\s.-]555[\s.-]01\d\d$/.test(m)],
    ["workers.dev hostname", /\b[a-z0-9-]+\.[a-z0-9-]+\.workers\.dev\b/gi, () => true],
    ["private key", new RegExp(["-----BEGIN", "PRIVATE KEY-----"].join(" (RSA |EC |OPENSSH )?"), "g"), () => true],
    ["API token-looking value", /\b(?:CLOUDFLARE_API_TOKEN|CLEF_LOOP_TOKEN)\s*[=:]\s*["']?[A-Za-z0-9_-]{20,}/g, () => true],
  ];

  it("has no ids, real contact details, hostnames or secrets in tracked files", async () => {
    const problems: string[] = [];
    for (const file of await walk(".")) {
      const text = await readFile(file, "utf8").catch(() => "");
      for (const [what, re, bad] of checks) {
        for (const m of text.match(re) ?? []) if (bad(m)) problems.push(`${file}: ${what}: ${m.slice(0, 40)}`);
      }
    }
    assert.deepEqual(problems, []);
  });
});
