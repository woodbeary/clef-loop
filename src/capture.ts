// Step 1: capture labeled decisions, from a CSV export or from AI Gateway logs.
//
// AI Gateway path (docs: ai-gateway/evaluations/add-human-feedback-api and the
// API reference for AI Gateway logs, checked 2026-10-31):
//   GET /accounts/{account}/ai-gateway/gateways/{gateway}/logs            list, per_page <= 50
//   GET /accounts/{account}/ai-gateway/gateways/{gateway}/logs/{id}/request
//   GET /accounts/{account}/ai-gateway/gateways/{gateway}/logs/{id}/response
// Labels are not on the logs. The canary Worker stores each label (a person's final
// call or a validator's verdict, with decided_by and decided_at) by request id, and the
// exporter joins it to the log whose `req` metadata matches. A log is exported only when
// its `tenant` metadata names a tenant in the consent file and the log falls inside that
// tenant's consent window, so a withdrawn or missing consent keeps a row out of every
// later build. Logs tagged `clef_loop_eval` are eval traffic and are skipped, so eval
// calls never leak into training data.

import { parseCsv } from "./csv.ts";
import type { Label } from "./labels.ts";
import { decisionFromSystemOne, isObject, validateDecision, validateQuestions } from "./schema.ts";
import type { Decision, DecisionRecord, Json, Questions } from "./types.ts";

export interface CaptureResult {
  records: DecisionRecord[];
  /** Rows or logs that were skipped, with the reason. */
  skipped: Array<{ id: string; reason: string }>;
  /** Logs with no label yet (not an error: most decisions are never labeled). */
  unlabeled?: number;
}

/** Who opted in to capture, and when. A tenant's records count only inside [from, until). */
export interface ConsentFile {
  tenants: Record<string, { from: string; until?: string | null }>;
}

export function parseConsent(text: string): ConsentFile {
  const v = JSON.parse(text) as unknown;
  if (!isObject(v) || !isObject(v.tenants)) throw new Error('consent file must look like {"tenants": {"<tenant>": {"from": "<ISO time>"}}}');
  for (const [t, c] of Object.entries(v.tenants)) {
    if (!isObject(c) || typeof c.from !== "string" || Number.isNaN(Date.parse(c.from))) throw new Error(`consent for "${t}" needs a "from" time`);
    if (c.until !== undefined && c.until !== null && (typeof c.until !== "string" || Number.isNaN(Date.parse(c.until)))) {
      throw new Error(`consent for "${t}" has an invalid "until" time`);
    }
  }
  return v as unknown as ConsentFile;
}

/** Null when the tenant's consent covers a record made at `at`; otherwise why not. */
export function consentProblem(consent: ConsentFile, tenant: string | undefined, at: string): string | null {
  if (!tenant) return "no tenant, so no consent on file";
  const c = consent.tenants[tenant];
  if (!c) return `tenant "${tenant}" has no consent on file`;
  const t = Date.parse(at);
  if (t < Date.parse(c.from)) return `made before tenant "${tenant}" opted in`;
  if (c.until && t >= Date.parse(c.until)) return `made after tenant "${tenant}" withdrew consent`;
  return null;
}

/** Keeps only records a tenant's consent covers; the rest are reported as skipped. */
export function applyConsent(result: CaptureResult, consent: ConsentFile): CaptureResult {
  const records: DecisionRecord[] = [];
  const skipped = [...result.skipped];
  for (const r of result.records) {
    const problem = consentProblem(consent, r.tenant, r.createdAt);
    if (problem) skipped.push({ id: r.id, reason: problem });
    else records.push(r);
  }
  return { ...result, records, skipped };
}

/**
 * The consent gate for a CSV export. With a consent file, only rows it covers are kept.
 * Without one, rows that name a tenant are refused outright: they are someone's
 * captured decisions, so they need a consent file. A CSV with no tenant column is the
 * team's own decision log and passes through.
 */
export function csvConsent(result: CaptureResult, consent: ConsentFile | undefined): CaptureResult {
  if (consent) return applyConsent(result, consent);
  const tenants = new Set(result.records.flatMap((r) => (r.tenant ? [r.tenant] : [])));
  if (tenants.size) {
    throw new Error(`${tenants.size} tenant(s) named in the CSV and no --consent FILE given: rows that name a tenant are exported only under a consent file`);
  }
  return result;
}

/**
 * CSV columns: id, created_at, state, final, decided_by, decided_at, and optionally
 * saw_suggestion (required when decided_by is human), model_output and tenant.
 * `state` may be text or JSON. `final` and `model_output` may be a JSON object of
 * question id -> option, or a bare option when there is exactly one question.
 * decided_by is human or validator; rows labeled by an agent are skipped.
 */
export function captureFromCsv(text: string, questions: Questions): CaptureResult {
  const qProblems = validateQuestions(questions);
  if (qProblems.length) throw new Error(`invalid questions: ${qProblems.join("; ")}`);
  const rows = parseCsv(text);
  const required = ["id", "created_at", "state", "final", "decided_by", "decided_at"];
  if (rows.length > 0) {
    const missing = required.filter((c) => !(c in rows[0]));
    if (missing.length) throw new Error(`CSV is missing columns: ${missing.join(", ")}`);
  }
  const records: DecisionRecord[] = [];
  const skipped: CaptureResult["skipped"] = [];
  rows.forEach((row, i) => {
    const id = row.id || `row-${i + 2}`;
    try {
      if (Number.isNaN(Date.parse(row.created_at))) throw new Error(`created_at "${row.created_at}" is not a date`);
      const final = parseDecision(row.final, questions, "final");
      const modelOutput = row.model_output ? parseDecision(row.model_output, questions, "model_output") : undefined;
      const who = decider(row.decided_by, row.decided_at, row.saw_suggestion === undefined || row.saw_suggestion === "" ? undefined : row.saw_suggestion);
      records.push({
        id,
        createdAt: new Date(row.created_at).toISOString(),
        state: parseMaybeJson(row.state),
        questions,
        final,
        ...who,
        ...(row.tenant ? { tenant: row.tenant } : {}),
        ...(modelOutput ? { modelOutput } : {}),
        source: "csv",
      });
    } catch (e) {
      skipped.push({ id, reason: (e as Error).message });
    }
  });
  return { records, skipped };
}

/** decided_by, decided_at and saw_suggestion, checked. Throws for agent labels and missing fields. */
export function decider(by: unknown, at: unknown, saw: unknown): Pick<DecisionRecord, "decidedBy" | "decidedAt" | "sawSuggestion"> {
  if (by === "agent") throw new Error("agent labels are refused: the label must be a person's final call or a validator's verdict");
  if (by !== "human" && by !== "validator") throw new Error(`decided_by must be human or validator, got ${JSON.stringify(by ?? null)}`);
  if (typeof at !== "string" || Number.isNaN(Date.parse(at))) throw new Error("decided_at must be an ISO 8601 time");
  const sawSuggestion = saw === true || saw === "true" ? true : saw === false || saw === "false" ? false : undefined;
  if (by === "human" && sawSuggestion === undefined) throw new Error("saw_suggestion (true or false) is required for a human label");
  if (by === "validator" && sawSuggestion === true) throw new Error("a validator must not see the model's answer");
  return { decidedBy: by, decidedAt: new Date(at).toISOString(), sawSuggestion: sawSuggestion ?? false };
}

/** Where labels come from: the canary Worker's GET /labels, or a fake in tests. */
export type LabelSource = (tenant: string) => Promise<Map<string, Label>>;

/** Reads every label for a tenant from the canary Worker, page by page. */
export function workerLabels(opts: { url: string; token: string; fetch?: typeof fetch }): LabelSource {
  const f = opts.fetch ?? fetch;
  return async (tenant) => {
    const out = new Map<string, Label>();
    let after: string | null = null;
    for (;;) {
      const u = new URL("/labels", opts.url);
      u.searchParams.set("tenant", tenant);
      u.searchParams.set("limit", "1000");
      if (after) u.searchParams.set("after", after);
      const res = await f(u.toString(), { headers: { Authorization: `Bearer ${opts.token}` } });
      if (!res.ok) throw new Error(`GET /labels?tenant=${tenant} -> ${res.status}`);
      const page = (await res.json()) as { labels: Label[]; next: string | null };
      for (const l of page.labels) out.set(l.request_id, l);
      if (!page.next) return out;
      after = page.next;
    }
  };
}

export interface GatewayCaptureOptions {
  accountId: string;
  gatewayId: string;
  apiToken: string;
  /** Who opted in, and when. Logs outside a tenant's consent are never exported. */
  consent: ConsentFile;
  /** Labels by request id, per tenant (the canary Worker's GET /labels). */
  labels: LabelSource;
  fetch?: typeof fetch;
  /** Stop after this many labeled records. */
  limit?: number;
  /** Used when a logged request has no `questions` (it always should for Clef). */
  questions?: Questions;
  baseUrl?: string;
}

/** Preference among a request's logs: the served System One call first, the shadow call last. */
const ROUTE_ORDER = ["fallback", "base", "tuned", "shadow"];

export async function captureFromGateway(opts: GatewayCaptureOptions): Promise<CaptureResult> {
  const f = opts.fetch ?? fetch;
  const base = `${opts.baseUrl ?? "https://api.cloudflare.com/client/v4"}/accounts/${encodeURIComponent(
    opts.accountId,
  )}/ai-gateway/gateways/${encodeURIComponent(opts.gatewayId)}/logs`;
  const headers = { Authorization: `Bearer ${opts.apiToken}` };
  const get = async (url: string): Promise<unknown> => {
    const res = await f(url, { headers });
    if (!res.ok) throw new Error(`GET ${redactUrl(url)} -> ${res.status}`);
    return unwrapEnvelope(await res.json());
  };

  const byReq = new Map<string, Array<{ id: string; createdAt: string; tenant: string; route: string }>>();
  const skipped: CaptureResult["skipped"] = [];
  const perPage = 50;
  for (let page = 1; ; page++) {
    const listUrl = `${base}?page=${page}&per_page=${perPage}&order_by=created_at&order_by_direction=asc`;
    const res = await f(listUrl, { headers });
    if (!res.ok) throw new Error(`GET ${redactUrl(listUrl)} -> ${res.status}`);
    const body = (await res.json()) as { result?: unknown; result_info?: { total_count?: number } };
    const logs = Array.isArray(body.result) ? body.result : [];
    for (const log of logs) {
      if (!isObject(log) || typeof log.id !== "string") continue;
      const meta = parseMetadata(log.metadata);
      if (meta.clef_loop_eval) continue;
      const createdAt = new Date(String(log.created_at)).toISOString();
      const tenant = typeof meta.tenant === "string" ? meta.tenant : undefined;
      const problem = consentProblem(opts.consent, tenant, createdAt);
      if (problem) {
        skipped.push({ id: log.id, reason: problem });
        continue;
      }
      if (typeof meta.req !== "string") {
        skipped.push({ id: log.id, reason: "no req metadata, so no label can be joined" });
        continue;
      }
      const group = byReq.get(meta.req) ?? [];
      group.push({ id: log.id, createdAt, tenant: tenant!, route: typeof meta.route === "string" ? meta.route : "base" });
      byReq.set(meta.req, group);
    }
    const total = body.result_info?.total_count;
    if (logs.length < perPage || (typeof total === "number" && page * perPage >= total)) break;
  }

  const labelsByTenant = new Map<string, Map<string, Label>>();
  for (const group of byReq.values()) {
    const t = group[0].tenant;
    if (!labelsByTenant.has(t)) labelsByTenant.set(t, await opts.labels(t));
  }

  const records: DecisionRecord[] = [];
  let unlabeled = 0;
  for (const [req, group] of byReq) {
    const label = labelsByTenant.get(group[0].tenant)?.get(req);
    if (!label) {
      unlabeled++;
      continue;
    }
    group.sort((a, b) => ROUTE_ORDER.indexOf(a.route) - ROUTE_ORDER.indexOf(b.route));
    const reasons: string[] = [];
    for (const log of group) {
      try {
        const request = await get(`${base}/${encodeURIComponent(log.id)}/request`);
        const input = systemOneInput(request);
        const questions = (input.questions as unknown as Questions | undefined) ?? opts.questions;
        if (!questions) throw new Error("request has no questions and none were given");
        const qProblems = validateQuestions(questions);
        if (qProblems.length) throw new Error(`invalid questions: ${qProblems.join("; ")}`);
        let modelOutput: Decision | undefined;
        if (log.route !== "shadow") {
          try {
            modelOutput = decisionFromSystemOne(await get(`${base}/${encodeURIComponent(log.id)}/response`), questions);
          } catch {
            modelOutput = undefined;
          }
        }
        const final = parseDecision(JSON.stringify(label.final), questions, "label.final");
        records.push({
          id: log.id,
          createdAt: log.createdAt,
          state: (input.state ?? null) as Json,
          questions,
          final,
          ...decider(label.decided_by, label.decided_at, label.saw_suggestion),
          tenant: log.tenant,
          ...(modelOutput ? { modelOutput } : {}),
          source: "ai-gateway",
        });
        reasons.length = 0;
        break;
      } catch (e) {
        reasons.push(`${log.route}: ${(e as Error).message}`);
      }
    }
    if (reasons.length) skipped.push({ id: group[0].id, reason: reasons.join("; ") });
    if (opts.limit && records.length >= opts.limit) break;
  }
  records.sort((a, b) => (a.createdAt < b.createdAt ? -1 : a.createdAt > b.createdAt ? 1 : 0));
  return { records, skipped, unlabeled };
}

/** Accepts the bare System One body or the /ai/run envelope ({ model, input }). */
export function systemOneInput(request: unknown): Record<string, Json> {
  if (!isObject(request)) throw new Error("logged request is not a JSON object");
  if (isObject(request.input) && ("state" in request.input || "questions" in request.input)) return request.input;
  if ("state" in request || "questions" in request) return request;
  throw new Error("logged request has no state or questions");
}

/** The Cloudflare API wraps results as { success, result, errors }. The log bodies are
 * typed `unknown` in the API reference, so accept either the envelope or the raw body,
 * and a JSON string inside either. */
export function unwrapEnvelope(body: unknown): unknown {
  let v = body;
  if (isObject(v) && "success" in v && "result" in v) v = v.result;
  if (typeof v === "string") {
    try {
      v = JSON.parse(v);
    } catch {
      // leave as string
    }
  }
  return v;
}

export function parseMetadata(raw: unknown): Record<string, Json> {
  if (isObject(raw)) return raw;
  if (typeof raw !== "string" || raw === "") return {};
  try {
    const v = JSON.parse(raw);
    return isObject(v) ? v : {};
  } catch {
    return {};
  }
}

export function parseDecision(raw: string, questions: Questions, field: string): Decision {
  const ids = Object.keys(questions);
  let decision: Decision;
  const parsed = parseMaybeJson(raw.trim());
  if (isObject(parsed)) {
    decision = Object.fromEntries(Object.entries(parsed).map(([k, v]) => [k, String(v)]));
  } else if (ids.length === 1 && raw.trim() !== "") {
    decision = { [ids[0]]: String(parsed) };
  } else {
    throw new Error(`${field} must be a JSON object of question id -> option`);
  }
  const problems = validateDecision(decision, questions);
  if (problems.length) throw new Error(`${field}: ${problems.join("; ")}`);
  return decision;
}

function parseMaybeJson(s: string): Json {
  const t = s.trim();
  if (t.startsWith("{") || t.startsWith("[")) {
    try {
      return JSON.parse(t) as Json;
    } catch {
      return s;
    }
  }
  return s;
}

function redactUrl(url: string): string {
  return url.replace(/\/accounts\/[^/]+/, "/accounts/<account>");
}
