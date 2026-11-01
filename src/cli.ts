#!/usr/bin/env node
// clef-loop CLI. Run with Node 22.18+ (TypeScript runs directly, no build step).

import { createHash } from "node:crypto";
import { mkdir, readdir, readFile, stat, writeFile, open } from "node:fs/promises";
import { dirname, join } from "node:path";
import { parseArgs } from "node:util";
import { buildCall, isTemplate, type CallableArm } from "./arms.ts";
import { byomPlan, checkCog } from "./byom.ts";
import { captureFromCsv, captureFromGateway, type CaptureResult } from "./capture.ts";
import { restTransport } from "./cloudflare.ts";
import { build, readClefJsonl, redactRecords } from "./dataset.ts";
import { evaluate, parseArm, renderMarkdown } from "./evaluate.ts";
import { checkAdapter, executeUpload, LORA_BASES, setModelType, uploadPlan } from "./lora.ts";
import { validateQuestions } from "./schema.ts";
import type { Questions } from "./types.ts";

const USAGE = `clef-loop: capture labeled decisions, redact, split, evaluate, deploy.

  clef-loop audit  (--csv FILE --questions FILE | --gateway ID [--limit N]) [--names FILE] [--sample N]
  clef-loop build  (--csv FILE --questions FILE | --gateway ID [--limit N]) --out DIR
                   [--heldout N|FRACTION] [--template mistral|gemma|llama2]
                   [--clef-model @cf/cloudflare/clef-flash] [--names FILE] [--keep-keys a,b] [--allow-residual]
  clef-loop eval   --heldout DIR/clef/heldout.jsonl [--train DIR/clef/train.jsonl]
                   --arm NAME=KIND[:ARG] ... [--out DIR] [--concurrency 4] [--note TEXT] [--dry-run]
                   [--abstain-below P]   route answers with top probability < P to a person
                   kinds: replay | majority | systemone:<model> | systemone-url:<url>
                          | lora:<model>|<finetune>|<template> | chat:<model>
  clef-loop lora check DIR [--model @cf/mistral/mistral-7b-instruct-v0.2-lora] [--set-model-type TYPE]
  clef-loop lora upload DIR --model MODEL --name NAME [--execute]
  clef-loop byom check [DIR]

Environment: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, AI_GATEWAY_ID (default clef-loop),
SYSTEMONE_URL_TOKEN (optional, for systemone-url arms).`;

async function main(argv: string[]): Promise<number> {
  const [cmd, sub, ...rest] = argv;
  switch (cmd) {
    case "audit":
      return audit([sub, ...rest].filter((x) => x !== undefined));
    case "build":
      return buildCmd([sub, ...rest].filter((x) => x !== undefined));
    case "eval":
      return evalCmd([sub, ...rest].filter((x) => x !== undefined));
    case "lora":
      if (sub === "check") return loraCheck(rest);
      if (sub === "upload") return loraUpload(rest);
      break;
    case "byom":
      if (sub === "check") return byomCheck(rest);
      break;
  }
  console.log(USAGE);
  return cmd === undefined || cmd === "help" || cmd === "--help" ? 0 : 2;
}

const SOURCE_OPTIONS = {
  csv: { type: "string" },
  questions: { type: "string" },
  gateway: { type: "string" },
  limit: { type: "string" },
  names: { type: "string" },
  "keep-keys": { type: "string" },
} as const;

async function capture(values: Record<string, string | boolean | string[] | undefined>): Promise<CaptureResult> {
  const questions = values.questions ? (JSON.parse(await readFile(String(values.questions), "utf8")) as Questions) : undefined;
  if (questions) {
    const problems = validateQuestions(questions);
    if (problems.length) throw new Error(`--questions: ${problems.join("; ")}`);
  }
  if (values.csv) {
    if (!questions) throw new Error("--csv needs --questions (a JSON file of the question schema)");
    return captureFromCsv(await readFile(String(values.csv), "utf8"), questions);
  }
  if (values.gateway) {
    return captureFromGateway({
      accountId: requireEnv("CLOUDFLARE_ACCOUNT_ID"),
      apiToken: requireEnv("CLOUDFLARE_API_TOKEN"),
      gatewayId: String(values.gateway),
      limit: values.limit ? Number(values.limit) : undefined,
      questions,
    });
  }
  throw new Error("give --csv FILE --questions FILE, or --gateway ID");
}

async function redactOptions(values: Record<string, string | boolean | string[] | undefined>) {
  const names = values.names
    ? (await readFile(String(values.names), "utf8")).split(/\r?\n/).map((s) => s.trim()).filter(Boolean)
    : [];
  const keepKeys = values["keep-keys"] ? String(values["keep-keys"]).split(",").map((s) => s.trim()) : [];
  return { names, keepKeys };
}

async function audit(args: string[]): Promise<number> {
  const { values } = parseArgs({ args, options: { ...SOURCE_OPTIONS, sample: { type: "string" } } });
  const captured = await capture(values);
  const red = redactRecords(captured.records, await redactOptions(values));
  console.log(`records: ${red.records.length}, skipped: ${captured.skipped.length}`);
  for (const s of captured.skipped.slice(0, 10)) console.log(`  skipped ${s.id}: ${s.reason}`);
  console.log(`redactions: ${JSON.stringify(red.counts)}`);
  console.log(`residual findings: ${red.residual.length}`);
  for (const f of red.residual.slice(0, 20)) console.log(`  ${f.path} ${f.kind} ${f.sample}`);
  const n = Number(values.sample ?? "3");
  for (const r of red.records.slice(0, n)) console.log(`\n--- ${r.id}\n${JSON.stringify(r.state, null, 2)}`);
  return red.residual.length ? 1 : 0;
}

async function buildCmd(args: string[]): Promise<number> {
  const { values } = parseArgs({
    args,
    options: {
      ...SOURCE_OPTIONS,
      out: { type: "string" },
      heldout: { type: "string", default: "0.2" },
      template: { type: "string", default: "mistral" },
      "clef-model": { type: "string", default: "@cf/cloudflare/clef-flash" },
      "allow-residual": { type: "boolean", default: false },
    },
  });
  if (!values.out) throw new Error("--out DIR is required");
  if (!isTemplate(values.template)) throw new Error("--template must be mistral, gemma or llama2");
  const captured = await capture(values);
  for (const s of captured.skipped) console.error(`skipped ${s.id}: ${s.reason}`);
  const out = build(captured.records, {
    heldOut: Number(values.heldout),
    template: values.template,
    clefModel: values["clef-model"],
    redact: await redactOptions(values),
    allowResidual: values["allow-residual"],
  });
  const manifest = JSON.parse(out.files["manifest.json"]);
  manifest.files = {} as Record<string, string>;
  for (const [name, text] of Object.entries(out.files)) {
    if (name === "manifest.json") continue;
    manifest.files[name] = `sha256:${createHash("sha256").update(text).digest("hex")}`;
  }
  out.files["manifest.json"] = JSON.stringify(manifest, null, 2) + "\n";
  for (const [name, text] of Object.entries(out.files)) {
    const path = join(values.out, name);
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, text);
  }
  console.log(
    `wrote ${values.out}: train ${out.split.train.length}, held-out ${out.split.heldOut.length} (from ${out.split.cutoff}), ` +
      `duplicates removed ${out.split.duplicatesRemoved}, label conflicts ${out.split.conflicts.length}, ` +
      `redactions ${JSON.stringify(out.redactions)}`,
  );
  return 0;
}

async function evalCmd(args: string[]): Promise<number> {
  const { values } = parseArgs({
    args,
    options: {
      heldout: { type: "string" },
      train: { type: "string" },
      arm: { type: "string", multiple: true },
      out: { type: "string" },
      concurrency: { type: "string", default: "4" },
      note: { type: "string" },
      "run-id": { type: "string" },
      "dry-run": { type: "boolean", default: false },
      "abstain-below": { type: "string" },
    },
  });
  if (!values.heldout) throw new Error("--heldout FILE is required");
  const abstainBelow = values["abstain-below"] === undefined ? undefined : Number(values["abstain-below"]);
  if (abstainBelow !== undefined && !(abstainBelow >= 0 && abstainBelow <= 1)) throw new Error("--abstain-below must be a probability in [0, 1]");
  if (!values.arm?.length) throw new Error("give at least one --arm");
  const heldOut = readClefJsonl(await readFile(values.heldout, "utf8"));
  const train = values.train ? readClefJsonl(await readFile(values.train, "utf8")) : undefined;
  const arms = values.arm.map(parseArm);
  if (values["dry-run"]) {
    for (const a of arms) {
      if (a.spec.kind === "replay" || a.spec.kind === "majority") {
        console.log(`# ${a.name}: ${a.spec.kind} (no call)`);
        continue;
      }
      const call = buildCall(a.spec as CallableArm, heldOut[0].state, heldOut[0].questions);
      console.log(`# ${a.name}: first of ${heldOut.length} calls\n${JSON.stringify(call, null, 2)}`);
    }
    return 0;
  }
  const runId = values["run-id"] ?? new Date().toISOString().replace(/[:.]/g, "-");
  const needsAccount = arms.some((a) => ["systemone", "lora", "chat"].includes(a.spec.kind));
  const transport = restTransport({
    accountId: needsAccount ? requireEnv("CLOUDFLARE_ACCOUNT_ID") : process.env.CLOUDFLARE_ACCOUNT_ID,
    apiToken: needsAccount ? requireEnv("CLOUDFLARE_API_TOKEN") : process.env.CLOUDFLARE_API_TOKEN,
    gatewayId: process.env.AI_GATEWAY_ID ?? "clef-loop",
    urlToken: process.env.SYSTEMONE_URL_TOKEN,
  });
  const report = await evaluate(heldOut, arms, { transport, train, concurrency: Number(values.concurrency), runId, abstainBelow });
  const md = renderMarkdown(report, values.note);
  if (values.out) {
    await mkdir(values.out, { recursive: true });
    await writeFile(join(values.out, "report.md"), md);
    await writeFile(join(values.out, "report.json"), JSON.stringify(report, null, 2) + "\n");
  }
  console.log(md);
  return 0;
}

async function loraCheck(args: string[]): Promise<number> {
  const { values, positionals } = parseArgs({
    args,
    allowPositionals: true,
    options: { model: { type: "string" }, "set-model-type": { type: "string" } },
  });
  const dir = positionals[0];
  if (!dir) throw new Error("lora check DIR");
  const cfgPath = join(dir, "adapter_config.json");
  if (values["set-model-type"]) {
    await writeFile(cfgPath, setModelType(await readFile(cfgPath, "utf8"), values["set-model-type"]));
    console.log(`set model_type=${values["set-model-type"]} in ${cfgPath}`);
  }
  const names = await readdir(dir);
  const st = names.includes("adapter_model.safetensors") ? await stat(join(dir, "adapter_model.safetensors")) : undefined;
  let head: Uint8Array | undefined;
  if (st) {
    const fh = await open(join(dir, "adapter_model.safetensors"), "r");
    try {
      const len = Buffer.alloc(8);
      await fh.read(len, 0, 8, 0);
      const n = Math.min(Number(len.readBigUInt64LE(0)), 100_000_000);
      const buf = Buffer.alloc(Math.min(8 + n, st.size));
      await fh.read(buf, 0, buf.length, 0);
      head = buf;
    } finally {
      await fh.close();
    }
  }
  const res = checkAdapter(
    {
      names,
      configText: names.includes("adapter_config.json") ? await readFile(cfgPath, "utf8") : undefined,
      safetensors: st && head ? { head, size: st.size } : undefined,
    },
    values.model,
  );
  console.log(JSON.stringify(res.info));
  for (const w of res.warnings) console.log(`warning: ${w}`);
  for (const e of res.errors) console.log(`error: ${e}`);
  console.log(res.errors.length ? "not ready to upload" : "ready to upload");
  return res.errors.length ? 1 : 0;
}

async function loraUpload(args: string[]): Promise<number> {
  const { values, positionals } = parseArgs({
    args,
    allowPositionals: true,
    options: { model: { type: "string" }, name: { type: "string" }, execute: { type: "boolean", default: false } },
  });
  const dir = positionals[0];
  if (!dir || !values.model || !values.name) throw new Error("lora upload DIR --model MODEL --name NAME");
  if (!LORA_BASES[values.model]) throw new Error(`--model must be one of ${Object.keys(LORA_BASES).join(", ")}`);
  const plan = uploadPlan(values.model, values.name, dir);
  if (!values.execute) {
    console.log("Dry run. Either run:\n  " + plan.wrangler + "\nor these REST calls (with --execute):");
    for (const s of plan.rest) console.log(`  ${s.method} ${s.path} ${JSON.stringify(s.json ?? s.multipart)}`);
    console.log(`\nThen call it with --arm tuned=lora:${values.model}|${values.name}|${LORA_BASES[values.model].template}`);
    return 0;
  }
  const id = await executeUpload(plan, {
    accountId: requireEnv("CLOUDFLARE_ACCOUNT_ID"),
    apiToken: requireEnv("CLOUDFLARE_API_TOKEN"),
    readFile: async (name) => new Uint8Array(await readFile(join(dir, name))),
  });
  console.log(`created fine-tune ${values.name} (${id})`);
  return 0;
}

async function byomCheck(args: string[]): Promise<number> {
  const dir = args[0] ?? "byom";
  const read = async (n: string) => readFile(join(dir, n), "utf8").catch(() => undefined);
  const res = checkCog({ cogYaml: await read("cog.yaml"), predictPy: await read("predict.py"), requirements: await read("requirements.txt") });
  for (const w of res.warnings) console.log(`warning: ${w}`);
  for (const e of res.errors) console.log(`error: ${e}`);
  console.log(res.errors.length ? "Cog package has errors" : "Cog package looks complete. Plan (dry run, nothing is pushed):");
  if (!res.errors.length) for (const line of byomPlan("clef-loop-tuned")) console.log(`  ${line}`);
  return res.errors.length ? 1 : 0;
}

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`${name} is not set`);
  return v;
}

main(process.argv.slice(2)).then(
  (code) => process.exit(code),
  (e: Error) => {
    console.error(`clef-loop: ${e.message}`);
    process.exit(1);
  },
);
