// Labels: the decision a person or a validator finally made, keyed by the canary's
// request id. They are stored by the Worker (one Durable Object per tenant), never
// attached to a gateway log by log id, because env.AI.aiGatewayLogId holds only the most
// recent call's id and can belong to another request under concurrency. The exporter
// joins each label to its gateway log through the `req` metadata the Worker sets on
// every call.
//
// A label must say who decided (`decided_by`) and when (`decided_at`). Agent-made labels
// are refused: training on a model's own answers rewards it for agreeing with itself.
// A human label records whether the person saw the model's suggestion first; a validator
// never does.

import type { KvStorage } from "./route-book.ts";
import { isObject } from "./schema.ts";
import type { Decision } from "./types.ts";

export type DecidedBy = "human" | "validator";

export interface Label {
  request_id: string;
  tenant: string;
  final: Decision;
  decided_by: DecidedBy;
  decided_at: string;
  saw_suggestion: boolean;
  recorded_at: string;
}

export class LabelError extends Error {
  readonly status: number;
  constructor(message: string, status = 400) {
    super(message);
    this.status = status;
  }
}

const MAX_CLOCK_SKEW_MS = 5 * 60_000;

export function parseLabel(body: unknown, now: Date): Label {
  if (!isObject(body)) throw new LabelError("body must be a JSON object");
  const { request_id, tenant, final, decided_by, decided_at, saw_suggestion } = body;
  if (typeof request_id !== "string" || !request_id) throw new LabelError("request_id is required");
  if (typeof tenant !== "string" || !tenant) throw new LabelError("tenant is required");
  if (!isObject(final) || Object.keys(final).length === 0 || !Object.values(final).every((v) => typeof v === "string")) {
    throw new LabelError("final must be an object of question id -> option");
  }
  if (decided_by === "agent") {
    throw new LabelError("agent labels are refused: the label must be a person's final call or a validator's verdict", 422);
  }
  if (decided_by !== "human" && decided_by !== "validator") throw new LabelError('decided_by must be "human" or "validator"');
  if (typeof decided_at !== "string" || Number.isNaN(Date.parse(decided_at))) throw new LabelError("decided_at must be an ISO 8601 time");
  if (Date.parse(decided_at) > now.getTime() + MAX_CLOCK_SKEW_MS) throw new LabelError("decided_at is in the future");
  if (decided_by === "human" && typeof saw_suggestion !== "boolean") {
    throw new LabelError("saw_suggestion (true or false) is required for a human label: did the person see the model's answer first?");
  }
  if (decided_by === "validator" && saw_suggestion === true) throw new LabelError("a validator must not see the model's answer");
  return {
    request_id,
    tenant,
    final: final as Decision,
    decided_by,
    decided_at: new Date(decided_at).toISOString(),
    saw_suggestion: decided_by === "human" ? (saw_suggestion as boolean) : false,
    recorded_at: now.toISOString(),
  };
}

/** One tenant's labels. A human label replaces any earlier label for the request; a validator label never replaces a human one. */
export class LabelBook {
  private readonly storage: KvStorage;

  constructor(storage: KvStorage) {
    this.storage = storage;
  }

  async put(label: Label): Promise<{ stored: boolean; reason?: string }> {
    const existingKey = await this.storage.get<string>(`r:${label.request_id}`);
    if (existingKey) {
      const existing = await this.storage.get<Label>(existingKey);
      if (existing?.decided_by === "human" && label.decided_by === "validator") {
        return { stored: false, reason: "a human label already exists for this request" };
      }
      await this.storage.delete(existingKey);
    }
    const key = `l:${label.recorded_at}:${label.request_id}`;
    await this.storage.put(key, label);
    await this.storage.put(`r:${label.request_id}`, key);
    return { stored: true };
  }

  async get(requestId: string): Promise<Label | undefined> {
    const key = await this.storage.get<string>(`r:${requestId}`);
    return key ? this.storage.get<Label>(key) : undefined;
  }

  /** Labels in the order they were recorded. Pass the returned `next` as `after` for the following page. */
  async list(after: string | null, limit: number): Promise<{ labels: Label[]; next: string | null }> {
    const m = await this.storage.list<Label>({ prefix: "l:", ...(after ? { startAfter: after } : {}), limit: limit + 1 });
    const entries = [...m.entries()];
    const page = entries.slice(0, limit);
    return { labels: page.map(([, l]) => l), next: entries.length > limit ? page[page.length - 1][0] : null };
  }

  async clear(): Promise<void> {
    await this.storage.deleteAll();
  }
}
