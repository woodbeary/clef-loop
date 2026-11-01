// The canary reverting itself on validator rejects, run locally with no account and no
// model: base answers with the rule that labeled the synthetic fixtures, and "tuned" is a
// stub that always gives a well-formed answer ("accept") that is often wrong. Nothing
// malformed, no errors, no timeouts: only the validator's rejects can trip the revert.
//
//   node scripts/demo-revert.ts

import type { AddressInfo } from "node:net";
import { pathToFileURL } from "node:url";
import { bucket, parseRevertConfig } from "../src/canary.ts";
import { LabelBook } from "../src/labels.ts";
import { memoryStorage, RouteBook, type RouteStatus } from "../src/route-book.ts";
import { handle, type AiBinding, type Deps, type Vars } from "../worker/handler.ts";
import { createValidatorServer } from "../validator/server.ts";
import { humanRule, QUESTIONS, syntheticLeads } from "./synthetic.ts";

export const TUNED_MEANS =
  "In the live demo, 'tuned' means trained on the open Clef-flash weights and evaluated offline; it is not a BYOM deployment.";

const TOKEN = "demo-token-not-a-secret";

function answer(choice: string) {
  const probabilities = { accept: 0.05, skip: 0.05, needs_info: 0.05, [choice]: 0.9 };
  return { model: "clef-flash", answers: { decision: { type: "choice", choice, probabilities } } };
}

export async function runRevertDemo(requests = 150): Promise<{ lines: string[]; status: RouteStatus; tunedServed: number; afterRoute: string }> {
  const server = createValidatorServer();
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const { port } = server.address() as AddressInfo;
  const lines: string[] = [];
  try {
    const ai: AiBinding = {
      async run(model, input) {
        const state = (input as { state: Parameters<typeof humanRule>[0] }).state;
        return model === "@cf/cloudflare/clef-flash" ? answer(humanRule(state)) : answer("accept");
      },
    };
    const route = new RouteBook(memoryStorage(), parseRevertConfig({}));
    const books = new Map<string, LabelBook>();
    const pending: Promise<unknown>[] = [];
    const deps: Deps = {
      ai,
      route,
      labels: (t) => books.get(t) ?? books.set(t, new LabelBook(memoryStorage())).get(t)!,
      waitUntil: (p) => void pending.push(p),
      validator: {
        fetch: async (r) => fetch(`http://127.0.0.1:${port}${new URL(r.url).pathname}`, { method: "POST", headers: r.headers, body: await r.text() }),
      },
    };
    const settle = async () => {
      for (let seen = -1; seen !== pending.length; ) {
        seen = pending.length;
        await Promise.all(pending);
      }
    };
    const vars: Vars = {
      CLEF_LOOP_TOKEN: TOKEN,
      GATEWAY_ID: "clef-loop",
      BASE_ARM: JSON.stringify({ kind: "systemone", model: "@cf/cloudflare/clef-flash" }),
      TUNED_ARM: JSON.stringify({ kind: "systemone", model: "tuned-stub", selector: "clef-flash" }),
      MODE: "canary",
      CANARY_PERCENT: "50",
      ROUTE_VERSION: "demo",
    };
    const call = (path: string, body?: unknown) =>
      handle(
        new Request(`https://canary.local/${path}`, {
          method: body === undefined ? "GET" : "POST",
          headers: { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" },
          body: body === undefined ? undefined : JSON.stringify(body),
        }),
        vars,
        deps,
      );
    lines.push(`CANARY_PERCENT=50, revert rule: 95% Wilson lower bound of tuned failures above max(1%, base's rate), judged from 50 tuned requests.`);
    lines.push(`'tuned' here is a stub that always answers "accept", well-formed. ${TUNED_MEANS}`);
    const leads = syntheticLeads().slice(0, requests);
    let tunedServed = 0;
    let wasReverted = false;
    for (const [i, lead] of leads.entries()) {
      const r = (await (await call("decide", { key: `lead-${i}`, state: lead.state, questions: QUESTIONS })).json()) as { route: string };
      await settle();
      if (r.route === "tuned") tunedServed++;
      const s = await route.status();
      if (s.reverted && !wasReverted) {
        wasReverted = true;
        lines.push(`request ${i + 1}: GET /route -> reverted: true, "${s.reason}"`);
      }
    }
    const status = (await (await call("route")).json()) as RouteStatus;
    const tunedKey = leads.map((_, i) => `lead-${i}`).find((k) => bucket(k, vars.ROUTE_VERSION) < 5000)!;
    const after = (await (await call("decide", { key: tunedKey, state: leads[0].state, questions: QUESTIONS })).json()) as { route: string };
    lines.push(`tuned served ${tunedServed} of ${leads.length}; tuned failures ${status.tuned.failures} of ${status.tuned.n}, base failures ${status.base.failures} of ${status.base.n}`);
    lines.push(`next /decide for ${tunedKey}, a key in the tuned bucket -> route: "${after.route}"`);
    return { lines, status, tunedServed, afterRoute: after.route };
  } finally {
    server.close();
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const { lines } = await runRevertDemo();
  for (const l of lines) console.log(l);
}
