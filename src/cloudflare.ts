// REST transport for the eval CLI (Node). The canary Worker uses the AI binding
// instead; both send the same Call objects from arms.ts.
//
// Endpoints (developers.cloudflare.com/ai-gateway/usage/rest-api/, checked 2026-10-31):
//   POST /accounts/{account}/ai/run/{model}          Workers AI models (@cf/...)
//   POST /accounts/{account}/ai/v1/chat/completions  third-party models (openai/gpt-5.5)
// with `cf-aig-gateway-id` so every call is logged in AI Gateway, and
// `cf-aig-metadata` so eval traffic is tagged and never captured as training data.

import { unwrapEnvelope } from "./capture.ts";
import type { Call } from "./arms.ts";
import { isObject } from "./schema.ts";

export type Metadata = Record<string, string | number | boolean>;

export interface Transport {
  send(call: Call, metadata: Metadata): Promise<unknown>;
}

export interface RestOptions {
  accountId?: string;
  apiToken?: string;
  gatewayId: string;
  fetch?: typeof fetch;
  baseUrl?: string;
  timeoutMs?: number;
  /** Attempts per call, including the first. Retries only 429 and 5xx. */
  attempts?: number;
  /** Bearer token for a `systemone-url` arm, if your server wants one. */
  urlToken?: string;
  sleep?: (ms: number) => Promise<void>;
}

export function restTransport(opts: RestOptions): Transport {
  const f = opts.fetch ?? fetch;
  const base = opts.baseUrl ?? "https://api.cloudflare.com/client/v4";
  const attempts = opts.attempts ?? 3;
  const sleep = opts.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)));
  const needAccount = () => {
    if (!opts.accountId || !opts.apiToken) {
      throw new Error("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required for Workers AI and chat arms");
    }
    return { account: encodeURIComponent(opts.accountId), token: opts.apiToken };
  };

  async function post(url: string, headers: Record<string, string>, body: unknown): Promise<unknown> {
    let lastError = "";
    for (let attempt = 1; attempt <= attempts; attempt++) {
      const res = await f(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...headers },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(opts.timeoutMs ?? 30_000),
      });
      if (res.ok) {
        const json = await res.json();
        if (isObject(json) && json.success === false) {
          throw new Error(`API error: ${JSON.stringify(json.errors ?? json).slice(0, 200)}`);
        }
        return unwrapEnvelope(json);
      }
      lastError = `${res.status} ${(await res.text()).slice(0, 200)}`;
      if (res.status !== 429 && res.status < 500) break;
      if (attempt < attempts) await sleep(250 * 2 ** (attempt - 1));
    }
    throw new Error(`POST failed: ${lastError}`);
  }

  return {
    async send(call, metadata) {
      const meta = JSON.stringify(metadata);
      switch (call.transport) {
        case "workers-ai": {
          const { account, token } = needAccount();
          return post(
            `${base}/accounts/${account}/ai/run/${call.model}`,
            { Authorization: `Bearer ${token}`, "cf-aig-gateway-id": opts.gatewayId, "cf-aig-metadata": meta },
            call.input,
          );
        }
        case "chat": {
          const { account, token } = needAccount();
          return post(
            `${base}/accounts/${account}/ai/v1/chat/completions`,
            { Authorization: `Bearer ${token}`, "cf-aig-gateway-id": opts.gatewayId, "cf-aig-metadata": meta },
            call.body,
          );
        }
        case "url":
          return post(call.url, opts.urlToken ? { Authorization: `Bearer ${opts.urlToken}` } : {}, call.body);
      }
    },
  };
}
