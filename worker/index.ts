// clef-loop canary Worker: wiring only. The routing, fallback and feedback logic is
// in handler.ts (tested in Node); the revert rule is in src/canary.ts.

import { DurableObject } from "cloudflare:workers";
import { parseRevertConfig } from "../src/canary.ts";
import { RouteBook, type RouteStatus, type VerdictResult, type WindowRoute } from "../src/route-book.ts";
import { handle, type AiBinding, type Vars } from "./handler.ts";

export interface Env extends Vars {
  AI: Ai;
  ROUTE: DurableObjectNamespace<RouteState>;
  REVERT_WINDOW?: string;
  REVERT_MIN_SAMPLES?: string;
  REVERT_MAX_FAILURE_RATE?: string;
}

export default {
  async fetch(request, env, ctx): Promise<Response> {
    // One Durable Object per route version holds the revert window.
    const stub = env.ROUTE.get(env.ROUTE.idFromName(`route:${env.ROUTE_VERSION}`));
    return handle(request, env, {
      ai: env.AI as unknown as AiBinding,
      route: {
        admit: () => stub.admit(),
        record: (route, failed, key) => stub.record(route, failed, key),
        verdict: (key, ok) => stub.verdict(key, ok),
        status: () => stub.status(),
        reset: () => stub.reset(),
      },
      waitUntil: (p) => ctx.waitUntil(p),
      log: (event) => console.log(JSON.stringify(event)),
    });
  },
} satisfies ExportedHandler<Env>;

/** Single-threaded; RouteBook also serializes its own read-modify-write steps. */
export class RouteState extends DurableObject<Env> {
  private readonly book: RouteBook;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.book = new RouteBook(ctx.storage, parseRevertConfig(env));
  }

  async admit(): Promise<boolean> {
    return this.book.admit();
  }

  async record(route: WindowRoute, failed: boolean, key?: string): Promise<RouteStatus> {
    return this.book.record(route, failed, key);
  }

  async verdict(key: string, ok: boolean): Promise<VerdictResult> {
    return this.book.verdict(key, ok);
  }

  async status(): Promise<RouteStatus> {
    return this.book.status();
  }

  async reset(): Promise<RouteStatus> {
    return this.book.reset();
  }
}
