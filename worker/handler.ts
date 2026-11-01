// Request handling for the canary Worker, written against small interfaces so the
// same code runs in the Worker (worker/index.ts) and in Node tests with a fake AI
// binding.
//
// It is also the capture shim. Every state is redacted here, before env.AI.run, so the
// model serves on the same redacted state it would be trained on and raw PII never
// reaches AI Gateway (whose DLP can flag or block, not redact). The gateway logs a
// request only for a tenant listed in CAPTURE_TENANTS (collectLog: true per request),
// and only when the residual audit finds nothing; everyone else gets collectLog: false
// and a metadata-only event (pTop, margin, latency, a state hash) in Workers Logs.
//
//   POST   /decide          { key?, tenant?, state, questions }
//                           -> { decision, route, fallback, captured, answers, needs_person, request_id }
//   POST   /feedback        { request_id, tenant, final, decided_by, decided_at, saw_suggestion }
//                           -> 204; stores the label (opted-in tenants only; agent labels refused)
//   GET    /labels?tenant=  -> { labels, next }; the exporter joins them to gateway logs by request id
//   DELETE /labels?tenant=  -> deletes a tenant's labels, for a withdrawn consent
//   GET    /route           -> revert state
//   POST   /route/reset     -> clears a revert
//
// Every model call goes through AI Gateway (`gateway.id`), tagged with the route, the
// tenant and the request id (`req`). All routes need `Authorization: Bearer <CLEF_LOOP_TOKEN>`.

import { buildCall, parseAnswer, type CallableArm } from "../src/arms.ts";
import { confidence } from "../src/calibration.ts";
import { chooseRoute, type Mode } from "../src/canary.ts";
import { LabelError, parseLabel, type Label } from "../src/labels.ts";
import { auditResidual, redactJson, type RedactionCounts } from "../src/redact.ts";
import type { VerdictResult, WindowRoute } from "../src/route-book.ts";
import { isObject, validateQuestions, type Probabilities } from "../src/schema.ts";
import { canonical } from "../src/split.ts";
import type { ArmSpec, Decision, Json, Questions } from "../src/types.ts";

type Meta = Record<string, string | number | boolean>;

/** The part of the Workers AI binding this Worker uses (see ai-gateway/usage/worker-binding-methods).
 * Model ids are plain strings because a BYOM model's id isn't in the generated types. It never reads
 * env.AI.aiGatewayLogId: that holds only the most recent call's id, which can be another request's. */
export interface AiBinding {
  run(model: string, input: unknown, options?: { gateway?: { id: string; metadata?: Meta; collectLog?: boolean } }): Promise<unknown>;
}

/** One tenant's labels (src/labels.ts). In the Worker this is a Durable Object per tenant. */
export interface LabelStore {
  put(label: Label): Promise<{ stored: boolean; reason?: string }>;
  list(after: string | null, limit: number): Promise<{ labels: Label[]; next: string | null }>;
  clear(): Promise<void>;
}

/** Holds the route's revert window (src/route-book.ts). In the Worker this is a Durable Object. */
export interface RouteStore {
  admit(): Promise<boolean>;
  record(route: WindowRoute, failed: boolean, key?: string): Promise<unknown>;
  verdict(key: string, ok: boolean): Promise<VerdictResult>;
  status(): Promise<unknown>;
  reset(): Promise<unknown>;
}

export interface Vars {
  CLEF_LOOP_TOKEN?: string;
  GATEWAY_ID: string;
  BASE_ARM: string;
  TUNED_ARM?: string;
  MODE?: string;
  CANARY_PERCENT?: string;
  ROUTE_VERSION: string;
  TUNED_TIMEOUT_MS?: string;
  /** Comma-separated tenant ids that opted in to capture. Empty: AI Gateway logs nobody's requests. */
  CAPTURE_TENANTS?: string;
  /** Flag answers whose top probability is below this for a person to decide. */
  ABSTAIN_BELOW?: string;
  /** Comma-separated state keys that look sensitive (service_name, ip_range) but carry no PII, kept as-is. */
  REDACT_KEEP_KEYS?: string;
}

/** One line per /decide, for every tenant. It never holds the state, only a hash of the redacted state. */
export interface DecisionEvent {
  event: "clef_loop.decision";
  req: string;
  tenant: string;
  route: "tuned" | "base";
  route_version: string;
  captured: boolean;
  capture_skipped: "no consent" | "residual PII" | null;
  latency_ms: number;
  state_hash: string;
  redactions: number;
  answers: Record<string, { p_top: number; margin: number }>;
  needs_person: string[];
}

export interface Deps {
  ai: AiBinding;
  route: RouteStore;
  labels(tenant: string): LabelStore;
  waitUntil(p: Promise<unknown>): void;
  fetch?: typeof fetch;
  log?(event: DecisionEvent): void;
  /** Monotonic-enough milliseconds for latency. */
  now?(): number;
  /** Wall-clock time for labels. */
  date?(): Date;
}

const TENANT = /^[A-Za-z0-9_.-]{1,64}$/;

export function captureTenants(raw: string | undefined): Set<string> {
  return new Set((raw ?? "").split(",").map((t) => t.trim()).filter(Boolean));
}

/** First 16 hex characters of SHA-256 over the canonical JSON of the (redacted) state. */
export async function stateHash(state: Json): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonical(state)));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

export async function handle(request: Request, vars: Vars, deps: Deps): Promise<Response> {
  if (!authorized(request.headers.get("Authorization"), vars.CLEF_LOOP_TOKEN)) return json({ error: "unauthorized" }, 401);
  const path = new URL(request.url).pathname;
  try {
    if (request.method === "POST" && path === "/decide") return await decide(request, vars, deps);
    if (request.method === "POST" && path === "/feedback") return await feedback(request, vars, deps);
    if (path === "/labels" && (request.method === "GET" || request.method === "DELETE")) return await labels(request, deps);
    if (request.method === "GET" && path === "/route") return json(await deps.route.status());
    if (request.method === "POST" && path === "/route/reset") return json(await deps.route.reset());
    return json({ error: "not found" }, 404);
  } catch (e) {
    if (e instanceof BadRequest) return json({ error: e.message }, 400);
    if (e instanceof LabelError) return json({ error: e.message }, e.status);
    throw e;
  }
}

async function decide(request: Request, vars: Vars, deps: Deps): Promise<Response> {
  const body = await readJson(request);
  const problems = validateQuestions(body.questions);
  if (problems.length) throw new BadRequest(`invalid questions: ${problems.join("; ")}`);
  if (!("state" in body)) throw new BadRequest("state is required");
  if (body.tenant !== undefined && (typeof body.tenant !== "string" || !TENANT.test(body.tenant))) {
    throw new BadRequest("tenant must be 1-64 letters, digits, '_', '.' or '-'");
  }
  const questions = body.questions as unknown as Questions;
  const tenant = typeof body.tenant === "string" ? body.tenant : "default";
  const counts: RedactionCounts = {};
  const keepKeys = (vars.REDACT_KEEP_KEYS ?? "").split(",").map((k) => k.trim()).filter(Boolean);
  const state = redactJson(body.state as Json, { keepKeys }, counts);
  const consented = captureTenants(vars.CAPTURE_TENANTS).has(tenant);
  const residual = consented && auditResidual(state).length > 0;
  const captured = consented && !residual;
  const key = typeof body.key === "string" && body.key ? body.key : crypto.randomUUID();
  const requestId = crypto.randomUUID();
  const now = deps.now ?? (() => Date.now());
  const started = now();
  const abstainBelow = vars.ABSTAIN_BELOW ? Number(vars.ABSTAIN_BELOW) : null;
  const respond = async (served: { decision: Decision; probs?: Probabilities }, route: "tuned" | "base", fallback: string | null) => {
    const answers: DecisionEvent["answers"] = {};
    for (const [qid, p] of Object.entries(served.probs ?? {})) {
      const c = confidence(p);
      answers[qid] = { p_top: c.pTop, margin: c.margin };
    }
    const needsPerson = abstainBelow === null ? [] : Object.entries(answers).filter(([, a]) => a.p_top < abstainBelow).map(([q]) => q);
    deps.log?.({
      event: "clef_loop.decision",
      req: requestId,
      tenant,
      route,
      route_version: vars.ROUTE_VERSION,
      captured,
      capture_skipped: captured ? null : consented ? "residual PII" : "no consent",
      latency_ms: now() - started,
      state_hash: await stateHash(state),
      redactions: Object.values(counts).reduce((a, b) => a + b, 0),
      answers,
      needs_person: needsPerson,
    });
    return json({
      decision: served.decision,
      route,
      fallback,
      captured,
      answers,
      needs_person: needsPerson,
      request_id: requestId,
    });
  };

  const base = parseArmVar(vars.BASE_ARM, "BASE_ARM");
  if (!base) throw new Error("BASE_ARM must be set");
  const tuned = parseArmVar(vars.TUNED_ARM, "TUNED_ARM");
  const mode: Mode = vars.MODE === "shadow" || vars.MODE === "canary" ? vars.MODE : "off";
  const meta: Meta = { route_version: vars.ROUTE_VERSION, req: requestId, tenant };
  const gw = { collectLog: captured };

  let route = chooseRoute({
    mode,
    percent: Number(vars.CANARY_PERCENT || "0"),
    key,
    salt: vars.ROUTE_VERSION,
    tunedConfigured: tuned !== null,
    reverted: false,
  });
  // Only canary-bucket requests pay for the round trip to the route store.
  if (route === "tuned" && !(await deps.route.admit())) route = "base";

  let fallback: string | null = null;
  if (route === "tuned" && tuned) {
    const ms = Number(vars.TUNED_TIMEOUT_MS || "1500");
    const r = await withTimeout(runArm(deps, vars, tuned, state, questions, { ...meta, route: "tuned" }, gw), ms);
    deps.waitUntil(deps.route.record("tuned", !r.ok, r.ok ? requestId : undefined));
    if (r.ok) return respond(r, "tuned", null);
    fallback = r.error;
  }

  const r = await runArm(deps, vars, base, state, questions, { ...meta, route: fallback ? "fallback" : "base" }, gw);
  deps.waitUntil(deps.route.record("base", !r.ok, r.ok ? requestId : undefined));
  if (mode === "shadow" && tuned) {
    deps.waitUntil(
      runArm(deps, vars, tuned, state, questions, { ...meta, route: "shadow" }, gw).then((s) =>
        deps.route.record("tuned", !s.ok, s.ok ? `${requestId}:shadow` : undefined),
      ),
    );
  }
  if (!r.ok) return json({ error: `base model failed: ${r.error}`, request_id: requestId }, 502);
  return respond(r, "base", fallback);
}

async function feedback(request: Request, vars: Vars, deps: Deps): Promise<Response> {
  const label = parseLabel(await readJson(request), deps.date?.() ?? new Date());
  if (!captureTenants(vars.CAPTURE_TENANTS).has(label.tenant)) {
    return json({ error: "this tenant has not opted in to capture, so nothing was logged to label" }, 409);
  }
  const r = await deps.labels(label.tenant).put(label);
  if (!r.stored) return json({ error: r.reason }, 409);
  return new Response(null, { status: 204 });
}

async function labels(request: Request, deps: Deps): Promise<Response> {
  const url = new URL(request.url);
  const tenant = url.searchParams.get("tenant") ?? "";
  if (!TENANT.test(tenant)) throw new BadRequest("tenant is required");
  if (request.method === "DELETE") {
    await deps.labels(tenant).clear();
    return json({ tenant, deleted: true });
  }
  const limit = Math.min(1000, Math.max(1, Number(url.searchParams.get("limit") ?? "500") || 500));
  return json(await deps.labels(tenant).list(url.searchParams.get("after"), limit));
}

type ArmRun = { ok: true; decision: Decision; probs?: Probabilities } | { ok: false; error: string };

async function runArm(
  deps: Deps,
  vars: Vars,
  arm: CallableArm,
  state: Json,
  questions: Questions,
  metadata: Meta,
  gw: { collectLog: boolean },
): Promise<ArmRun> {
  try {
    const call = buildCall(arm, state, questions);
    const gateway = { id: vars.GATEWAY_ID, metadata, collectLog: gw.collectLog };
    let raw: unknown;
    if (call.transport === "workers-ai") {
      raw = await deps.ai.run(call.model, call.input, { gateway });
    } else if (call.transport === "chat") {
      const { model, ...input } = call.body;
      raw = await deps.ai.run(String(model), input, { gateway });
    } else {
      const res = await (deps.fetch ?? fetch)(call.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(call.body),
      });
      if (!res.ok) throw new Error(`${call.url} -> ${res.status}`);
      raw = await res.json();
    }
    const answer = parseAnswer(arm, raw, questions);
    return { ok: true, decision: answer.decision, ...(answer.probs ? { probs: answer.probs } : {}) };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
}

function withTimeout(p: Promise<ArmRun>, ms: number): Promise<ArmRun> {
  return new Promise((resolve) => {
    const t = setTimeout(() => resolve({ ok: false, error: `timed out after ${ms} ms` }), ms);
    p.then((r) => {
      clearTimeout(t);
      resolve(r);
    });
  });
}

export function parseArmVar(raw: string | undefined, name: string): CallableArm | null {
  if (!raw) return null;
  let spec: ArmSpec;
  try {
    spec = JSON.parse(raw) as ArmSpec;
  } catch {
    throw new Error(`${name} is not valid JSON`);
  }
  if (spec.kind === "replay" || spec.kind === "majority") throw new Error(`${name} must call a model, not ${spec.kind}`);
  return spec;
}

/** Constant-time comparison of the bearer token. */
export function authorized(header: string | null, token: string | undefined): boolean {
  if (!token || !header) return false;
  const given = new TextEncoder().encode(header.replace(/^Bearer\s+/i, ""));
  const want = new TextEncoder().encode(token);
  let diff = given.length ^ want.length;
  for (let i = 0; i < want.length; i++) diff |= (given[i] ?? 0) ^ want[i];
  return diff === 0;
}

async function readJson(request: Request): Promise<Record<string, Json>> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    throw new BadRequest("body must be JSON");
  }
  if (!isObject(body)) throw new BadRequest("body must be a JSON object");
  return body;
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });
}

class BadRequest extends Error {}
