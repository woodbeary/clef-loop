// Minimal repros for the gaps in docs/GAPS.md that a plain REST call shows. Dry run by
// default: prints each request with the account id elided and sends nothing. With
// --execute it sends them with CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN and prints
// each status and the first 300 characters of the reply, for a bug report.
//
//   node repros/rest.ts [clef-lora|context-24k|hf-files ...] [--execute]

import { pathToFileURL } from "node:url";

export interface Repro {
  name: string;
  gap: string;
  expect: string;
  method: "GET" | "POST";
  url: string;
  needsAccount: boolean;
  body?: unknown;
}

const QUESTIONS = {
  ok: { type: "noul", instructions: "Does the text describe a working checkout?" },
};

export function repros(): Repro[] {
  const run = (model: string) => `https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/${model}`;
  return [
    {
      name: "clef-lora",
      gap: "Workers AI takes LoRA adapters only for listed bases; Clef's input schema has no lora field.",
      expect: "A 4xx naming the unknown `lora` field, or a 200 that ignores it. Either shows there is no Clef LoRA path.",
      method: "POST",
      url: run("@cf/cloudflare/clef-flash"),
      needsAccount: true,
      body: { model: "clef-flash", state: "Checkout works.", questions: QUESTIONS, lora: "clef-loop-repro" },
    },
    {
      name: "context-24k",
      gap: "The launch changelog says 64K for Clef-flash; the model page says 24,576 tokens since Oct 9.",
      expect: "An error naming the 24,576-token context window for a state of about 30,000 tokens.",
      method: "POST",
      url: run("@cf/cloudflare/clef-flash"),
      needsAccount: true,
      body: { model: "clef-flash", state: "checkout ".repeat(30_000), questions: QUESTIONS },
    },
    {
      name: "hf-files",
      gap: "The Clef-flash weights ship inference code (joint_schema_model.py) and no training code.",
      expect: "A file list with joint_head.safetensors and joint_schema_model.py and no training script.",
      method: "GET",
      url: "https://huggingface.co/api/models/Cloudflare/clef-flash",
      needsAccount: false,
    },
  ];
}

export function describe(r: Repro): string {
  const body = r.body === undefined ? "" : ` ${JSON.stringify(r.body).slice(0, 160)}${JSON.stringify(r.body).length > 160 ? "..." : ""}`;
  return `# ${r.name}: ${r.gap}\n${r.method} ${r.url}${body}\nexpect: ${r.expect}`;
}

export async function execute(r: Repro, env: Record<string, string | undefined>, f: typeof fetch = fetch): Promise<string> {
  let url = r.url;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (r.needsAccount) {
    if (!env.CLOUDFLARE_ACCOUNT_ID || !env.CLOUDFLARE_API_TOKEN) throw new Error(`${r.name} needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN`);
    url = url.replace("{account}", encodeURIComponent(env.CLOUDFLARE_ACCOUNT_ID));
    headers.Authorization = `Bearer ${env.CLOUDFLARE_API_TOKEN}`;
  }
  const res = await f(url, { method: r.method, headers, body: r.body === undefined ? undefined : JSON.stringify(r.body) });
  const text = (await res.text()).slice(0, 300);
  return `${r.name}: ${res.status} ${text}`;
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const args = process.argv.slice(2);
  const go = args.includes("--execute");
  const names = args.filter((a) => !a.startsWith("--"));
  const chosen = repros().filter((r) => names.length === 0 || names.includes(r.name));
  for (const r of chosen) {
    console.log(describe(r));
    if (go) console.log(await execute(r, process.env));
    console.log("");
  }
  if (!go) console.log("dry run: nothing sent. Add --execute to send these with your account.");
}
