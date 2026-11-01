// Request handling for the canary Worker, written against small interfaces so the
// same code runs in the Worker (worker/index.ts) and in Node tests with a fake AI
// binding.
//
//   POST /decide        { key?, state, questions } -> { decision, route, fallback, log_id, request_id }
//   POST /feedback      { log_id, final, route }   -> 204; attaches the human's decision to the log
//   GET  /route                                    -> revert state
//   POST /route/reset                              -> clears a revert
//
// Every model call goes through AI Gateway (`gateway.id`), tagged with the route and
// a request id, so production traffic becomes the next training set. All routes need
// `Authorization: Bearer <CLEF_LOOP_TOKEN>`.

import { buildCall, parseResult, type CallableArm } from "../src/arms.ts";
import { chooseRoute, type Mode } from "../src/canary.ts";
import type { VerdictResult, WindowRoute } from "../src/route-book.ts";
import { isObject, validateQuestions } from "../src/schema.ts";
import type { ArmSpec, Decision, Json, Questions } from "../src/types.ts";

type Meta = Record<string, string | number | boolean>;

/** The parts of the Workers AI binding this Worker uses (see ai-gateway/usage/worker-binding-methods).
 * Model ids are plain strings because a BYOM model's id isn't in the generated types. */
export interface AiBinding {
  run(model: string, input: unknown, options?: { gateway?: { id: string; metadata?: Meta } }): Promise<unknown>;
  aiGatewayLogId: string | null;
  gateway(id: string): { patchLog(logId: string, data: { feedback?: -1 | 1; score?: number; metadata?: Meta }): Promise<void> };
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
}

export interface Deps {
  ai: AiBinding;
  route: RouteStore;
  waitUntil(p: Promise<unknown>): void;
  fetch?: typeof fetch;
}

export async function handle(request: Request, vars: Vars, deps: Deps): Promise<Response> {
  if (!authorized(request.headers.get("Authorization"), vars.CLEF_LOOP_TOKEN)) return json({ error: "unauthorized" }, 401);
  const path = new URL(request.url).pathname;
  try {
    if (request.method === "POST" && path === "/decide") return await decide(request, vars, deps);
    if (request.method === "POST" && path === "/feedback") return await feedback(request, vars, deps);
    if (request.method === "GET" && path === "/route") return json(await deps.route.status());
    if (request.method === "POST" && path === "/route/reset") return json(await deps.route.reset());
    return json({ error: "not found" }, 404);
  } catch (e) {
    if (e instanceof BadRequest) return json({ error: e.message }, 400);
    throw e;
  }
}

async function decide(request: Request, vars: Vars, deps: Deps): Promise<Response> {
  const body = await readJson(request);
  const problems = validateQuestions(body.questions);
  if (problems.length) throw new BadRequest(`invalid questions: ${problems.join("; ")}`);
  if (!("state" in body)) throw new BadRequest("state is required");
  const questions = body.questions as unknown as Questions;
  const state = body.state as Json;
  const key = typeof body.key === "string" && body.key ? body.key : crypto.randomUUID();
  const requestId = crypto.randomUUID();

  const base = parseArmVar(vars.BASE_ARM, "BASE_ARM");
  if (!base) throw new Error("BASE_ARM must be set");
  const tuned = parseArmVar(vars.TUNED_ARM, "TUNED_ARM");
  const mode: Mode = vars.MODE === "shadow" || vars.MODE === "canary" ? vars.MODE : "off";
  const meta: Meta = { route_version: vars.ROUTE_VERSION, req: requestId };

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
    const r = await withTimeout(runArm(deps, vars, tuned, state, questions, { ...meta, route: "tuned" }), ms);
    deps.waitUntil(deps.route.record("tuned", !r.ok, r.ok ? requestId : undefined));
    if (r.ok) return json({ decision: r.decision, route: "tuned", fallback: null, log_id: r.logId, request_id: requestId });
    fallback = r.error;
  }

  const r = await runArm(deps, vars, base, state, questions, { ...meta, route: fallback ? "fallback" : "base" });
  deps.waitUntil(deps.route.record("base", !r.ok, r.ok ? requestId : undefined));
  // The shadow call starts after the base call has returned and its log id has been read.
  // env.AI.aiGatewayLogId holds only the last call's id, so running both at once could hand
  // the caller the shadow log's id, and /feedback would label the wrong log.
  if (mode === "shadow" && tuned) {
    deps.waitUntil(
      runArm(deps, vars, tuned, state, questions, { ...meta, route: "shadow" }).then((s) =>
        deps.route.record("tuned", !s.ok, s.ok ? `${requestId}:shadow` : undefined),
      ),
    );
  }
  if (!r.ok) return json({ error: `base model failed: ${r.error}`, request_id: requestId }, 502);
  return json({ decision: r.decision, route: "base", fallback, log_id: r.logId, request_id: requestId });
}

async function feedback(request: Request, vars: Vars, deps: Deps): Promise<Response> {
  const body = await readJson(request);
  if (typeof body.log_id !== "string" || !body.log_id) throw new BadRequest("log_id is required");
  if (!isObject(body.final) || !Object.values(body.final).every((v) => typeof v === "string")) {
    throw new BadRequest("final must be an object of question id -> option");
  }
  // patchLog's merge behavior for metadata isn't documented, so the route is sent
  // again with the label. The log stays usable whether it merges or replaces.
  await deps.ai.gateway(vars.GATEWAY_ID).patchLog(body.log_id, {
    metadata: {
      final: JSON.stringify(body.final),
      route: typeof body.route === "string" ? body.route : "unknown",
      route_version: vars.ROUTE_VERSION,
    },
  });
  return new Response(null, { status: 204 });
}

type ArmRun = { ok: true; decision: Decision; logId: string | null } | { ok: false; error: string };

async function runArm(deps: Deps, vars: Vars, arm: CallableArm, state: Json, questions: Questions, metadata: Meta): Promise<ArmRun> {
  try {
    const call = buildCall(arm, state, questions);
    const gateway = { id: vars.GATEWAY_ID, metadata };
    let raw: unknown;
    let logId: string | null = null;
    if (call.transport === "workers-ai") {
      raw = await deps.ai.run(call.model, call.input, { gateway });
      // Documented to hold the last call's log id; the `req` metadata finds the log if not.
      logId = deps.ai.aiGatewayLogId;
    } else if (call.transport === "chat") {
      const { model, ...input } = call.body;
      raw = await deps.ai.run(String(model), input, { gateway });
      logId = deps.ai.aiGatewayLogId;
    } else {
      const res = await (deps.fetch ?? fetch)(call.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(call.body),
      });
      if (!res.ok) throw new Error(`${call.url} -> ${res.status}`);
      raw = await res.json();
    }
    return { ok: true, decision: parseResult(arm, raw, questions), logId };
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
