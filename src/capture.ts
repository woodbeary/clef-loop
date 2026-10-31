// Step 1: capture labeled decisions, from a CSV export or from AI Gateway logs.
//
// AI Gateway path (docs: ai-gateway/evaluations/add-human-feedback-api and the
// API reference for AI Gateway logs, checked 2026-10-31):
//   GET /accounts/{account}/ai-gateway/gateways/{gateway}/logs            list, per_page <= 50
//   GET /accounts/{account}/ai-gateway/gateways/{gateway}/logs/{id}/request
//   GET /accounts/{account}/ai-gateway/gateways/{gateway}/logs/{id}/response
// The human's final decision is attached to the log as custom metadata under
// `final` (the canary Worker's /feedback route does this with patchLog). Logs
// tagged `clef_loop_eval` are eval traffic and are skipped, so eval calls never
// leak into training data.

import { parseCsv } from "./csv.ts";
import { decisionFromSystemOne, isObject, validateDecision, validateQuestions } from "./schema.ts";
import type { Decision, DecisionRecord, Json, Questions } from "./types.ts";

export interface CaptureResult {
  records: DecisionRecord[];
  /** Rows or logs that were skipped, with the reason. */
  skipped: Array<{ id: string; reason: string }>;
}

/**
 * CSV columns: id, created_at, state, final, and optionally model_output.
 * `state` may be text or JSON. `final` and `model_output` may be a JSON object of
 * question id -> option, or a bare option when there is exactly one question.
 */
export function captureFromCsv(text: string, questions: Questions): CaptureResult {
  const qProblems = validateQuestions(questions);
  if (qProblems.length) throw new Error(`invalid questions: ${qProblems.join("; ")}`);
  const rows = parseCsv(text);
  const required = ["id", "created_at", "state", "final"];
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
      records.push({
        id,
        createdAt: new Date(row.created_at).toISOString(),
        state: parseMaybeJson(row.state),
        questions,
        final,
        ...(modelOutput ? { modelOutput } : {}),
        source: "csv",
      });
    } catch (e) {
      skipped.push({ id, reason: (e as Error).message });
    }
  });
  return { records, skipped };
}

export interface GatewayCaptureOptions {
  accountId: string;
  gatewayId: string;
  apiToken: string;
  fetch?: typeof fetch;
  /** Stop after this many labeled records. */
  limit?: number;
  /** Used when a logged request has no `questions` (it always should for Clef). */
  questions?: Questions;
  baseUrl?: string;
}

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

  const records: DecisionRecord[] = [];
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
      if (meta.final === undefined) continue;
      try {
        const request = await get(`${base}/${encodeURIComponent(log.id)}/request`);
        const response = await get(`${base}/${encodeURIComponent(log.id)}/response`);
        const input = systemOneInput(request);
        const questions = (input.questions as unknown as Questions | undefined) ?? opts.questions;
        if (!questions) throw new Error("request has no questions and none were given");
        const qProblems = validateQuestions(questions);
        if (qProblems.length) throw new Error(`invalid questions: ${qProblems.join("; ")}`);
        let modelOutput: Decision | undefined;
        try {
          modelOutput = decisionFromSystemOne(response, questions);
        } catch {
          modelOutput = undefined; // Not a System One response; keep the label without it.
        }
        records.push({
          id: log.id,
          createdAt: new Date(String(log.created_at)).toISOString(),
          state: (input.state ?? null) as Json,
          questions,
          final: parseDecision(String(meta.final), questions, "metadata.final"),
          ...(modelOutput ? { modelOutput } : {}),
          source: "ai-gateway",
        });
      } catch (e) {
        skipped.push({ id: log.id, reason: (e as Error).message });
      }
      if (opts.limit && records.length >= opts.limit) return { records, skipped };
    }
    const total = body.result_info?.total_count;
    if (logs.length < perPage || (typeof total === "number" && page * perPage >= total)) break;
  }
  return { records, skipped };
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
