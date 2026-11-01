import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { promisify } from "node:util";
import {
  applyConsent,
  captureFromCsv,
  captureFromGateway,
  csvConsent,
  parseConsent,
  parseDecision,
  systemOneInput,
  unwrapEnvelope,
  workerLabels,
  type ConsentFile,
} from "../src/capture.ts";
import type { Label } from "../src/labels.ts";
import { QUESTIONS } from "../scripts/synthetic.ts";

describe("captureFromCsv", () => {
  it("reads the synthetic fixture", async () => {
    const csv = await readFile("fixtures/leads.csv", "utf8");
    const { records, skipped } = captureFromCsv(csv, QUESTIONS);
    assert.equal(skipped.length, 0);
    assert.equal(records.length, 303);
    const first = records[0];
    assert.equal(first.id, "lead-0001");
    assert.equal(typeof first.state, "object");
    assert.ok(first.final.decision);
    assert.ok(first.modelOutput?.decision);
  });

  it("skips bad rows with a reason instead of failing the whole file", () => {
    const csv = [
      "id,created_at,state,final,model_output,decided_by,decided_at,saw_suggestion",
      'ok,2026-09-01T00:00:00Z,"{""a"":1}",accept,skip,human,2026-09-01T00:10:00Z,true',
      "bad-date,yesterday,text,accept,,human,2026-09-01T00:10:00Z,false",
      "bad-label,2026-09-01T00:00:00Z,text,maybe,,human,2026-09-01T00:10:00Z,false",
      'json-label,2026-09-02T00:00:00Z,plain text,"{""decision"":""skip""}",,validator,2026-09-02T00:01:00Z,',
    ].join("\n");
    const { records, skipped } = captureFromCsv(csv, QUESTIONS);
    assert.deepEqual(records.map((r) => r.id), ["ok", "json-label"]);
    assert.deepEqual(records[0].state, { a: 1 });
    assert.deepEqual([records[0].decidedBy, records[0].decidedAt, records[0].sawSuggestion], ["human", "2026-09-01T00:10:00.000Z", true]);
    assert.equal(records[1].state, "plain text");
    assert.deepEqual([records[1].decidedBy, records[1].sawSuggestion], ["validator", false]);
    assert.deepEqual(skipped.map((s) => s.id), ["bad-date", "bad-label"]);
    assert.match(skipped[1].reason, /not one of/);
  });

  it("requires the core columns, including who decided and when", () => {
    assert.throws(() => captureFromCsv("id,state\n1,x\n", QUESTIONS), /missing columns: created_at, final, decided_by, decided_at/);
    assert.throws(() => captureFromCsv("id,created_at,state,final\n1,2026-09-01T00:00:00Z,x,skip\n", QUESTIONS), /missing columns: decided_by, decided_at/);
  });

  it("refuses agent-made labels and human labels that don't say whether the suggestion was seen", () => {
    const csv = [
      "id,created_at,state,final,decided_by,decided_at,saw_suggestion",
      "agent,2026-09-01T00:00:00Z,x,skip,agent,2026-09-01T00:01:00Z,false",
      "unsaid,2026-09-01T00:00:00Z,y,skip,human,2026-09-01T00:01:00Z,",
      "nobody,2026-09-01T00:00:00Z,z,skip,,2026-09-01T00:01:00Z,false",
      "when,2026-09-01T00:00:00Z,w,skip,human,,false",
      "fine,2026-09-01T00:00:00Z,v,skip,human,2026-09-01T00:01:00Z,false",
    ].join("\n");
    const { records, skipped } = captureFromCsv(csv, QUESTIONS);
    assert.deepEqual(records.map((r) => r.id), ["fine"]);
    assert.match(skipped.find((s) => s.id === "agent")!.reason, /agent labels are refused/);
    assert.match(skipped.find((s) => s.id === "unsaid")!.reason, /saw_suggestion/);
    assert.match(skipped.find((s) => s.id === "nobody")!.reason, /decided_by/);
    assert.match(skipped.find((s) => s.id === "when")!.reason, /decided_at/);
  });

  it("keeps only rows a tenant's consent covers when a consent file is given", () => {
    const csv = [
      "id,created_at,state,final,decided_by,decided_at,saw_suggestion,tenant",
      "a,2026-09-10T00:00:00Z,x,skip,human,2026-09-10T00:01:00Z,false,acme",
      "early,2026-08-01T00:00:00Z,y,skip,human,2026-08-01T00:01:00Z,false,acme",
      "late,2026-10-20T00:00:00Z,z,skip,human,2026-10-20T00:01:00Z,false,acme",
      "other,2026-09-10T00:00:00Z,w,skip,human,2026-09-10T00:01:00Z,false,trial",
      "none,2026-09-10T00:00:00Z,v,skip,human,2026-09-10T00:01:00Z,false,",
    ].join("\n");
    const consent = parseConsent(JSON.stringify({ tenants: { acme: { from: "2026-09-01T00:00:00Z", until: "2026-10-01T00:00:00Z" } } }));
    const out = applyConsent(captureFromCsv(csv, QUESTIONS), consent);
    assert.deepEqual(out.records.map((r) => r.id), ["a"]);
    const why = Object.fromEntries(out.skipped.map((s) => [s.id, s.reason]));
    assert.match(why.early, /before tenant "acme" opted in/);
    assert.match(why.late, /after tenant "acme" withdrew/);
    assert.match(why.other, /"trial" has no consent on file/);
    assert.match(why.none, /no tenant/);
    assert.throws(() => parseConsent('{"acme":{}}'), /tenants/);
    assert.throws(() => parseConsent('{"tenants":{"acme":{}}}'), /needs a "from" time/);
  });

  it("refuses a CSV whose rows name a tenant unless a consent file is given", () => {
    const header = "id,created_at,state,final,decided_by,decided_at,saw_suggestion,tenant";
    const tenanted = [header, "a,2026-09-10T00:00:00Z,x,skip,human,2026-09-10T00:01:00Z,false,acme"].join("\n");
    assert.throws(() => csvConsent(captureFromCsv(tenanted, QUESTIONS), undefined), /no --consent FILE/);
    const consent = parseConsent(JSON.stringify({ tenants: { acme: { from: "2026-09-01T00:00:00Z" } } }));
    assert.deepEqual(csvConsent(captureFromCsv(tenanted, QUESTIONS), consent).records.map((r) => r.id), ["a"]);
    const own = [header, "b,2026-09-10T00:00:00Z,y,skip,human,2026-09-10T00:01:00Z,false,"].join("\n");
    assert.deepEqual(csvConsent(captureFromCsv(own, QUESTIONS), undefined).records.map((r) => r.id), ["b"]);
  });
});

describe("clef-loop build --csv (the CLI)", () => {
  it("refuses a CSV whose rows name a tenant without --consent, and builds it with one", async () => {
    const dir = await mkdtemp(join(tmpdir(), "clef-loop-cli-"));
    try {
      const [header, ...rows] = (await readFile("fixtures/leads.csv", "utf8")).trim().split("\n");
      await writeFile(join(dir, "t.csv"), [`${header},tenant`, ...rows.slice(0, 40).map((r) => `${r},acme`)].join("\n"));
      await writeFile(join(dir, "consent.json"), JSON.stringify({ tenants: { acme: { from: "2026-01-01T00:00:00Z" } } }));
      const cli = (...extra: string[]) =>
        promisify(execFile)(process.execPath, ["src/cli.ts", "build", "--csv", join(dir, "t.csv"), "--questions", "fixtures/questions.json", "--names", "fixtures/names.txt", ...extra]);
      await assert.rejects(cli("--out", join(dir, "refused")), (e: { code?: number; stderr?: string }) => e.code === 1 && /no --consent FILE given/.test(e.stderr ?? ""));
      const ok = await cli("--consent", join(dir, "consent.json"), "--out", join(dir, "built"));
      assert.match(ok.stdout, /train 32, held-out 8/);
      const manifest = JSON.parse(await readFile(join(dir, "built", "manifest.json"), "utf8"));
      assert.deepEqual(manifest.tenants, { acme: 40 });
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe("parseDecision", () => {
  it("accepts a bare option only when there is one question", () => {
    assert.deepEqual(parseDecision("skip", QUESTIONS, "final"), { decision: "skip" });
    const two = { ...QUESTIONS, urgent: { type: "noul" as const, instructions: "Urgent?" } };
    assert.throws(() => parseDecision("skip", two, "final"), /JSON object/);
    assert.deepEqual(parseDecision('{"decision":"skip","urgent":"true"}', two, "final"), { decision: "skip", urgent: "true" });
  });
});

describe("captureFromGateway", () => {
  const consent: ConsentFile = { tenants: { acme: { from: "2026-09-01T00:30:00Z" } } };
  const at = (i: number) => new Date(Date.UTC(2026, 8, 1, i)).toISOString();
  const human = (req: string, decision: string, saw = false): Label => ({
    request_id: req,
    tenant: "acme",
    final: { decision },
    decided_by: "human",
    decided_at: at(60),
    saw_suggestion: saw,
    recorded_at: at(60),
  });

  // A fake of the documented AI Gateway logs API (list, paged, per_page <= 50, then
  // /request and /response per log), with the metadata the canary Worker sets.
  function fakeApi() {
    const special: Record<number, Record<string, unknown>> = {
      1: { req: "r1", tenant: "acme", route: "base", clef_loop_eval: true },
      2: { req: "r2", tenant: "acme", route: "base" },
      3: { req: "r3", tenant: "trial", route: "base" },
      4: { route: "base", tenant: "acme" },
      5: { req: "r5", tenant: "acme", route: "tuned" },
      6: { req: "r5", tenant: "acme", route: "fallback" },
      7: { req: "r7", tenant: "acme", route: "shadow" },
      8: { req: "r7", tenant: "acme", route: "base" },
      9: { req: "r9", tenant: "acme", route: "base" },
    };
    const logs = Array.from({ length: 63 }, (_, i) => ({
      id: `log-${i}`,
      created_at: at(i),
      metadata: JSON.stringify(special[i] ?? (i === 0 ? { req: "r0", tenant: "acme", route: "base" } : { req: `r${i}`, tenant: "acme", route: "base" })),
    }));
    const calls: string[] = [];
    const f = (async (url: string, init?: RequestInit) => {
      calls.push(url);
      assert.equal((init?.headers as Record<string, string>).Authorization, "Bearer t0ken");
      const u = new URL(url);
      const m = u.pathname.match(/\/logs\/([^/]+)\/(request|response)$/);
      if (m) {
        const i = Number(m[1].split("-")[1]);
        if (m[2] === "request") {
          if (i === 5) return Response.json({ prompt: "<s>[INST] ... [/INST]", raw: true, lora: "leads-v1" });
          const body = { model: "clef-flash", state: { n: i }, questions: QUESTIONS };
          return Response.json(i % 3 === 0 ? { success: true, result: { model: "@cf/cloudflare/clef-flash", input: body } } : body);
        }
        const choice = i === 7 ? "needs_info" : "accept";
        const answers = { decision: { type: "choice", choice, probabilities: { accept: 0.8, skip: 0.1, needs_info: 0.1 } } };
        return Response.json({ success: true, result: JSON.stringify({ model: "clef-flash", answers }) });
      }
      const page = Number(u.searchParams.get("page"));
      const per = Number(u.searchParams.get("per_page"));
      assert.ok(per <= 50);
      assert.equal(u.searchParams.get("order_by"), "created_at");
      return Response.json({ success: true, result: logs.slice((page - 1) * per, page * per), result_info: { total_count: logs.length } });
    }) as typeof fetch;
    const labels = new Map<string, Label>();
    for (let i = 0; i < 63; i++) if (i !== 2) labels.set(`r${i}`, human(`r${i}`, i % 2 ? "accept" : "skip", i % 4 === 0));
    labels.set("r9", { ...human("r9", "skip"), decided_by: "agent" as never });
    const asked: string[] = [];
    const source = async (tenant: string) => {
      asked.push(tenant);
      return tenant === "acme" ? labels : new Map<string, Label>();
    };
    return { f, calls, source, asked };
  }

  it("joins labels by request id, exports only consented tenants, and refuses agent labels", async () => {
    const { f, calls, source, asked } = fakeApi();
    const { records, skipped, unlabeled } = await captureFromGateway({ accountId: "acct", gatewayId: "gw", apiToken: "t0ken", fetch: f, consent, labels: source });
    const ids = records.map((r) => r.id);
    for (const gone of ["log-0", "log-1", "log-2", "log-3", "log-4", "log-5", "log-7", "log-9"]) assert.ok(!ids.includes(gone), `${gone} was exported`);
    assert.equal(unlabeled, 1);
    const why = Object.fromEntries(skipped.map((s) => [s.id, s.reason]));
    assert.match(why["log-0"], /before tenant "acme" opted in/);
    assert.match(why["log-3"], /"trial" has no consent on file/);
    assert.match(why["log-4"], /no req metadata/);
    assert.match(why["log-9"], /agent labels are refused/);
    assert.ok(!("log-1" in why), "eval traffic is skipped silently");
    const fallback = records.find((r) => r.id === "log-6")!;
    assert.deepEqual(fallback.state, { n: 6 });
    assert.deepEqual(fallback.final, { decision: "accept" });
    const base = records.find((r) => r.id === "log-8")!;
    assert.deepEqual(base.modelOutput, { decision: "accept" });
    assert.equal(records.filter((r) => r.state && (r.state as { n: number }).n === 7).length, 0);
    assert.equal(records.length, 55);
    assert.ok(records.every((r) => r.tenant === "acme" && r.decidedBy === "human" && r.source === "ai-gateway"));
    assert.deepEqual(records.find((r) => r.id === "log-12")!.sawSuggestion, true);
    assert.deepEqual(asked, ["acme"]);
    assert.equal(calls.filter((c) => c.includes("page=")).length, 2);
    assert.ok(!calls.some((c) => c.includes("log-3/")), "a log without consent was read");
  });

  it("stops at --limit", async () => {
    const { f, source } = fakeApi();
    const { records } = await captureFromGateway({ accountId: "acct", gatewayId: "gw", apiToken: "t0ken", fetch: f, limit: 5, consent, labels: source });
    assert.equal(records.length, 5);
  });

  it("reports API errors without the account id", async () => {
    const f = (async () => new Response("nope", { status: 403 })) as unknown as typeof fetch;
    await assert.rejects(
      captureFromGateway({ accountId: "secret-account", gatewayId: "gw", apiToken: "t", fetch: f, consent, labels: async () => new Map() }),
      (e: Error) => e.message.includes("403") && !e.message.includes("secret-account"),
    );
  });
});

describe("workerLabels", () => {
  it("pages through the canary Worker's GET /labels with the bearer token", async () => {
    const seen: string[] = [];
    const f = (async (url: string, init?: RequestInit) => {
      seen.push(url);
      assert.equal((init?.headers as Record<string, string>).Authorization, "Bearer tok");
      const after = new URL(url).searchParams.get("after");
      const mk = (id: string) => ({ request_id: id, tenant: "acme", final: { decision: "skip" }, decided_by: "human", decided_at: "2026-10-01T00:00:00.000Z", saw_suggestion: false, recorded_at: "2026-10-01T00:00:00.000Z" });
      return Response.json(after ? { labels: [mk("c")], next: null } : { labels: [mk("a"), mk("b")], next: "cursor-1" });
    }) as typeof fetch;
    const labels = await workerLabels({ url: "https://canary.example", token: "tok", fetch: f })("acme");
    assert.deepEqual([...labels.keys()], ["a", "b", "c"]);
    assert.match(seen[0], /\/labels\?tenant=acme&limit=1000$/);
    assert.match(seen[1], /after=cursor-1/);
  });
});

describe("log body helpers", () => {
  it("unwraps the API envelope and JSON strings", () => {
    assert.deepEqual(unwrapEnvelope({ success: true, result: '{"a":1}' }), { a: 1 });
    assert.deepEqual(unwrapEnvelope({ a: 1 }), { a: 1 });
  });

  it("finds state and questions in either request shape", () => {
    assert.deepEqual(systemOneInput({ input: { state: 1, questions: {} } }), { state: 1, questions: {} });
    assert.deepEqual(systemOneInput({ state: 1, questions: {} }), { state: 1, questions: {} });
    assert.throws(() => systemOneInput({ messages: [] }), /no state or questions/);
    assert.throws(() => systemOneInput({ prompt: "<s>[INST] ... [/INST]", raw: true, lora: "leads-v1" }), /no state or questions/);
  });
});
