// Shared by the Worker tests: the handler's dependencies with a fake AI binding, the
// real RouteBook and LabelBook over in-memory storage, and request builders.

import { parseRevertConfig } from "../src/canary.ts";
import { LabelBook } from "../src/labels.ts";
import { memoryStorage, RouteBook, type RouteStatus } from "../src/route-book.ts";
import type { AiBinding, DecisionEvent, Deps, RouteStore, Vars } from "../worker/handler.ts";
import { QUESTIONS } from "../scripts/synthetic.ts";

export const TOKEN = "test-token-not-a-secret";
const LORA = { kind: "lora", model: "@cf/mistral/mistral-7b-instruct-v0.2-lora", lora: "leads-v1", template: "mistral" };

export function vars(over: Partial<Vars> = {}): Vars {
  return {
    CLEF_LOOP_TOKEN: TOKEN,
    GATEWAY_ID: "clef-loop",
    BASE_ARM: JSON.stringify({ kind: "systemone", model: "@cf/cloudflare/clef-flash" }),
    TUNED_ARM: JSON.stringify(LORA),
    MODE: "canary",
    CANARY_PERCENT: "100",
    ROUTE_VERSION: "v1",
    TUNED_TIMEOUT_MS: "50",
    ...over,
  };
}

export type TunedBehavior = "ok" | "error" | "prose" | "slow";

export function harness(tuned: TunedBehavior = "ok") {
  const runs: Array<{ model: string; input: Record<string, unknown>; metadata: Record<string, unknown>; collectLog?: boolean }> = [];
  const ai: AiBinding & { aiGatewayLogId?: string } = {
    get aiGatewayLogId(): string {
      throw new Error("the Worker read env.AI.aiGatewayLogId, which can hold another request's log id");
    },
    async run(model, input, options) {
      runs.push({ model, input: input as Record<string, unknown>, metadata: options?.gateway?.metadata ?? {}, collectLog: options?.gateway?.collectLog });
      if (model === "@cf/cloudflare/clef-flash") {
        return { model: "clef-flash", answers: { decision: { type: "choice", choice: "skip", probabilities: { accept: 0.1, skip: 0.8, needs_info: 0.1 }, confidence: 0.7 } }, usage: { input_tokens: 1, output_tokens: 0 } };
      }
      if (tuned === "error") throw new Error("5007: model unavailable");
      if (tuned === "prose") return { response: "Probably accept, it looks fine" };
      if (tuned === "slow") await new Promise((r) => setTimeout(r, 200));
      return { response: "accept" };
    },
  };
  const route: RouteStore = new RouteBook(memoryStorage(), parseRevertConfig({ REVERT_MIN_SAMPLES: "5", REVERT_WINDOW: "20" }));
  const pending: Promise<unknown>[] = [];
  const events: DecisionEvent[] = [];
  let clock = 1000;
  const books = new Map<string, LabelBook>();
  const labels = (tenant: string) => books.get(tenant) ?? books.set(tenant, new LabelBook(memoryStorage())).get(tenant)!;
  const deps: Deps = { ai, route, labels, waitUntil: (p) => void pending.push(p), log: (e) => void events.push(e), now: () => (clock += 7) };
  const settle = async () => {
    for (let seen = -1; seen !== pending.length; ) {
      seen = pending.length;
      await Promise.all(pending);
    }
  };
  return { deps, runs, events, labels, settle, status: () => route.status() as Promise<RouteStatus> };
}

export function req(path: string, body?: unknown, token = TOKEN): Request {
  return new Request(`https://canary.example/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export const decideBody = { key: "lead-1", state: { service: "yard debris", distance_miles: 4 }, questions: QUESTIONS };

