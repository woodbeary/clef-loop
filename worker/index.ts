// clef-loop canary Worker: wiring only. The routing, fallback and feedback logic is
// in handler.ts (tested in Node); the revert rule is in src/canary.ts.

import { DurableObject } from "cloudflare:workers";
import { EMPTY_STATE, parseRevertConfig, recordOutcome, type RevertState } from "../src/canary.ts";
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
        record: (failed) => stub.record(failed),
        status: () => stub.status(),
        reset: () => stub.reset(),
      },
      waitUntil: (p) => ctx.waitUntil(p),
    });
  },
} satisfies ExportedHandler<Env>;

/** Single-threaded, and record() awaits only storage between its read and write, so updates can't interleave. */
export class RouteState extends DurableObject<Env> {
  async admit(): Promise<boolean> {
    return !(await this.load()).reverted;
  }

  async record(failed: boolean): Promise<{ reverted: boolean; reason: string | null }> {
    const next = recordOutcome(await this.load(), failed, parseRevertConfig(this.env), new Date());
    await this.ctx.storage.put("state", next);
    return { reverted: next.reverted, reason: next.reason };
  }

  async status(): Promise<{ reverted: boolean; reason: string | null; revertedAt: string | null; window: number; failures: number }> {
    const s = await this.load();
    return { reverted: s.reverted, reason: s.reason, revertedAt: s.revertedAt, window: s.recent.length, failures: s.recent.filter(Boolean).length };
  }

  async reset(): Promise<{ reverted: false }> {
    await this.ctx.storage.put("state", EMPTY_STATE);
    return { reverted: false };
  }

  private async load(): Promise<RevertState> {
    return (await this.ctx.storage.get<RevertState>("state")) ?? EMPTY_STATE;
  }
}
