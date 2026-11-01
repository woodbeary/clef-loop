// The canary's revert window, written against a small key-value storage interface so
// the same code runs in the Durable Object (ctx.storage) and in Node tests (memoryStorage).
//
// Each request adds one entry to its route's window: failed if the call errored, timed
// out or returned an answer that doesn't fit the questions. A later verdict (a validator
// reject through POST /outcome, or the Worker's own validator call) can turn that entry
// into a failure, so a well-formed wrong answer counts once, as a failure, not as one
// success plus one failure. Entries older than the window are dropped, and a verdict for
// a dropped request is not counted.

import { judge, type Judgement, type RevertConfig } from "./canary.ts";

export interface KvStorage {
  get<T>(key: string): Promise<T | undefined>;
  put<T>(key: string, value: T): Promise<void>;
  delete(key: string): Promise<boolean>;
  deleteAll(): Promise<void>;
  list<T>(options: { prefix: string; reverse?: boolean; limit?: number; startAfter?: string }): Promise<Map<string, T>>;
}

export type WindowRoute = "tuned" | "base";

interface Entry {
  failed: boolean;
  key: string | null;
}

interface Index {
  route: WindowRoute;
  seq: number;
}

interface Revert {
  reverted: boolean;
  reason: string | null;
  revertedAt: string | null;
}

export interface RouteStatus extends Revert {
  limit: number;
  tuned: Judgement["tuned"];
  base: Judgement["base"];
}

export interface VerdictResult {
  counted: boolean;
  route: WindowRoute | null;
  reason?: string;
}

const NOT_REVERTED: Revert = { reverted: false, reason: null, revertedAt: null };
const pad = (n: number) => String(n).padStart(12, "0");

export class RouteBook {
  private readonly storage: KvStorage;
  private readonly cfg: RevertConfig;
  private readonly now: () => Date;
  private queue: Promise<unknown> = Promise.resolve();

  constructor(storage: KvStorage, cfg: RevertConfig, now: () => Date = () => new Date()) {
    this.storage = storage;
    this.cfg = cfg;
    this.now = now;
  }

  async admit(): Promise<boolean> {
    return !(await this.revertState()).reverted;
  }

  /** Adds one request to a route's window. `key` lets a later verdict find it; omit it for entries no verdict can change. */
  record(route: WindowRoute, failed: boolean, key?: string): Promise<RouteStatus> {
    return this.serial(() => this.recordNow(route, failed, key));
  }

  /** A validator's verdict on a recorded request. A reject turns its entry into a failure; repeating it changes nothing. */
  verdict(key: string, ok: boolean): Promise<VerdictResult> {
    return this.serial(() => this.verdictNow(key, ok));
  }

  reset(): Promise<RouteStatus> {
    return this.serial(async () => {
      await this.storage.deleteAll();
      return this.status();
    });
  }

  async status(): Promise<RouteStatus> {
    const j = judge(await this.window("tuned"), await this.window("base"), this.cfg);
    return { ...(await this.revertState()), limit: j.limit, tuned: j.tuned, base: j.base };
  }

  private serial<T>(f: () => Promise<T>): Promise<T> {
    const run = this.queue.then(f, f);
    this.queue = run.catch(() => undefined);
    return run;
  }

  private async recordNow(route: WindowRoute, failed: boolean, key?: string): Promise<RouteStatus> {
    const seq = ((await this.storage.get<number>(`seq:${route}`)) ?? 0) + 1;
    await this.storage.put(`seq:${route}`, seq);
    await this.storage.put<Entry>(`e:${route}:${pad(seq)}`, { failed, key: key ?? null });
    if (key) await this.storage.put<Index>(`id:${key}`, { route, seq });
    const old = seq - this.cfg.window;
    if (old > 0) {
      const gone = await this.storage.get<Entry>(`e:${route}:${pad(old)}`);
      await this.storage.delete(`e:${route}:${pad(old)}`);
      if (gone?.key) await this.storage.delete(`id:${gone.key}`);
    }
    return this.rejudge();
  }

  private async verdictNow(key: string, ok: boolean): Promise<VerdictResult> {
    const idx = await this.storage.get<Index>(`id:${key}`);
    if (!idx) return { counted: false, route: null, reason: "unknown request, or older than the revert window" };
    const entryKey = `e:${idx.route}:${pad(idx.seq)}`;
    const entry = await this.storage.get<Entry>(entryKey);
    if (!entry) return { counted: false, route: idx.route, reason: "older than the revert window" };
    if (entry.failed || ok) return { counted: !entry.failed && ok, route: idx.route };
    await this.storage.put<Entry>(entryKey, { ...entry, failed: true });
    await this.rejudge();
    return { counted: true, route: idx.route };
  }

  private async rejudge(): Promise<RouteStatus> {
    const current = await this.revertState();
    const j = judge(await this.window("tuned"), await this.window("base"), this.cfg);
    let next = current;
    if (!current.reverted && j.revert) {
      next = { reverted: true, reason: j.reason, revertedAt: this.now().toISOString() };
      await this.storage.put<Revert>("revert", next);
    }
    return { ...next, limit: j.limit, tuned: j.tuned, base: j.base };
  }

  private async window(route: WindowRoute): Promise<boolean[]> {
    const m = await this.storage.list<Entry>({ prefix: `e:${route}:`, reverse: true, limit: this.cfg.window });
    return [...m.values()].reverse().map((e) => e.failed);
  }

  private async revertState(): Promise<Revert> {
    return (await this.storage.get<Revert>("revert")) ?? NOT_REVERTED;
  }
}

/** An in-memory KvStorage with the Durable Object list semantics clef-loop uses (sorted keys, prefix, reverse, limit). */
export function memoryStorage(): KvStorage & { size(): number } {
  const m = new Map<string, unknown>();
  return {
    async get<T>(key: string) {
      return structuredClone(m.get(key)) as T | undefined;
    },
    async put<T>(key: string, value: T) {
      m.set(key, structuredClone(value));
    },
    async delete(key: string) {
      return m.delete(key);
    },
    async deleteAll() {
      m.clear();
    },
    async list<T>(o: { prefix: string; reverse?: boolean; limit?: number; startAfter?: string }) {
      let keys = [...m.keys()].filter((k) => k.startsWith(o.prefix) && (o.startAfter === undefined || k > o.startAfter)).sort();
      if (o.reverse) keys.reverse();
      if (o.limit !== undefined) keys = keys.slice(0, o.limit);
      return new Map(keys.map((k) => [k, structuredClone(m.get(k)) as T]));
    },
    size: () => m.size,
  };
}
