// Step 5, documented path: Workers AI LoRA fine-tunes.
//
// Rules from developers.cloudflare.com/workers-ai/features/fine-tunes/loras/ and the
// AutoTrain tutorial (both last updated 2026-04-21, checked 2026-10-31):
// - files must be named exactly adapter_config.json and adapter_model.safetensors
// - adapter_model.safetensors must be under 300 MB
// - rank r <= 8 is the documented default; ranks up to 32 are also accepted
// - adapter_config.json must include model_type: mistral, gemma or llama
// - the base model must be one Workers AI serves with LoRA, and not quantized
// - upload: `wrangler ai finetune create <model> <name> <folder>` or the REST API
//   POST /accounts/{id}/ai/finetunes, then POST .../finetunes/{id}/finetune-assets
//   once per file (multipart: file_name, file)

import { isObject } from "./schema.ts";
import type { TemplateName } from "./types.ts";

export const LORA_BASES: Record<string, { hf: string; modelType: "mistral" | "gemma" | "llama"; template: TemplateName }> = {
  "@cf/mistral/mistral-7b-instruct-v0.2-lora": { hf: "mistralai/Mistral-7B-Instruct-v0.2", modelType: "mistral", template: "mistral" },
  "@cf/google/gemma-2b-it-lora": { hf: "google/gemma-2b-it", modelType: "gemma", template: "gemma" },
  "@cf/google/gemma-7b-it-lora": { hf: "google/gemma-7b-it", modelType: "gemma", template: "gemma" },
  "@cf/meta-llama/llama-2-7b-chat-hf-lora": { hf: "meta-llama/Llama-2-7b-chat-hf", modelType: "llama", template: "llama2" },
};

export const MAX_ADAPTER_BYTES = 300_000_000;
const MODEL_TYPES = ["mistral", "gemma", "llama"];

export interface AdapterFiles {
  /** File names present in the folder. */
  names: string[];
  configText?: string;
  /** First bytes of adapter_model.safetensors (enough for the header) and its full size. */
  safetensors?: { head: Uint8Array; size: number };
}

export interface CheckResult {
  errors: string[];
  warnings: string[];
  info: { rank?: number; modelType?: string; baseModel?: string; tensors?: number; dtypes?: string[] };
}

export function checkAdapter(files: AdapterFiles, workersAiModel?: string): CheckResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  const info: CheckResult["info"] = {};
  for (const required of ["adapter_config.json", "adapter_model.safetensors"]) {
    if (!files.names.includes(required)) errors.push(`missing ${required} (the name must match exactly)`);
  }
  const extra = files.names.filter((n) => !["adapter_config.json", "adapter_model.safetensors"].includes(n));
  if (extra.length) warnings.push(`only adapter_config.json and adapter_model.safetensors are uploaded; ignoring ${extra.join(", ")}`);

  if (files.configText !== undefined) {
    let cfg: unknown;
    try {
      cfg = JSON.parse(files.configText);
    } catch {
      errors.push("adapter_config.json is not valid JSON");
    }
    if (isObject(cfg)) {
      const r = cfg.r;
      if (typeof r !== "number") errors.push("adapter_config.json has no numeric r (rank)");
      else {
        info.rank = r;
        if (r > 32) errors.push(`rank r=${r} is above 32, which Workers AI does not accept`);
        else if (r > 8) warnings.push(`rank r=${r}: the docs name r <= 8 first and also accept up to 32; test inference after upload`);
      }
      const mt = cfg.model_type;
      if (typeof mt !== "string") errors.push('adapter_config.json needs "model_type" (mistral, gemma or llama); run with --set-model-type');
      else {
        info.modelType = mt;
        if (!MODEL_TYPES.includes(mt)) errors.push(`model_type "${mt}" must be one of ${MODEL_TYPES.join(", ")}`);
      }
      if (typeof cfg.task_type === "string" && cfg.task_type !== "CAUSAL_LM") warnings.push(`task_type is ${cfg.task_type}, expected CAUSAL_LM`);
      if (typeof cfg.base_model_name_or_path === "string") info.baseModel = cfg.base_model_name_or_path;
      if (workersAiModel) {
        const base = LORA_BASES[workersAiModel];
        if (!base) errors.push(`${workersAiModel} is not one of the LoRA base models clef-loop knows: ${Object.keys(LORA_BASES).join(", ")}`);
        else {
          if (typeof mt === "string" && mt !== base.modelType) errors.push(`model_type "${mt}" does not match ${workersAiModel} (${base.modelType})`);
          if (info.baseModel && info.baseModel !== base.hf) warnings.push(`trained on ${info.baseModel}; ${workersAiModel} serves ${base.hf}`);
        }
      }
    }
  }

  if (files.safetensors) {
    if (files.safetensors.size >= MAX_ADAPTER_BYTES) {
      errors.push(`adapter_model.safetensors is ${files.safetensors.size} bytes; the limit is under 300 MB`);
    }
    try {
      const header = readSafetensorsHeader(files.safetensors.head, files.safetensors.size);
      const tensors = Object.entries(header).filter(([k]) => k !== "__metadata__");
      info.tensors = tensors.length;
      info.dtypes = [...new Set(tensors.map(([, v]) => (isObject(v) ? String(v.dtype) : "?")))];
      if (tensors.length === 0) errors.push("adapter_model.safetensors has no tensors");
      const quant = info.dtypes.filter((d) => /^(I8|U8|I4|U4|F8_E4M3|F8_E5M2)$/.test(d));
      if (quant.length) errors.push(`adapter has quantized tensors (${quant.join(", ")}); Workers AI needs an unquantized adapter`);
    } catch (e) {
      errors.push(`adapter_model.safetensors: ${(e as Error).message}`);
    }
  }
  return { errors, warnings, info };
}

/** safetensors layout: 8-byte little-endian header length, then that many bytes of JSON. */
export function readSafetensorsHeader(head: Uint8Array, fileSize: number): Record<string, unknown> {
  if (head.byteLength < 8) throw new Error("file is too short to be safetensors");
  const n = Number(new DataView(head.buffer, head.byteOffset, 8).getBigUint64(0, true));
  if (n <= 0 || n > 100_000_000 || 8 + n > fileSize) throw new Error(`bad header length ${n}`);
  if (head.byteLength < 8 + n) throw new Error(`need ${8 + n} bytes to read the header, got ${head.byteLength}`);
  const header = JSON.parse(new TextDecoder().decode(head.subarray(8, 8 + n)));
  if (!isObject(header)) throw new Error("header is not a JSON object");
  return header;
}

export function setModelType(configText: string, modelType: string): string {
  if (!MODEL_TYPES.includes(modelType)) throw new Error(`model_type must be one of ${MODEL_TYPES.join(", ")}`);
  const cfg = JSON.parse(configText);
  if (!isObject(cfg)) throw new Error("adapter_config.json is not a JSON object");
  return JSON.stringify({ ...cfg, model_type: modelType }, null, 2) + "\n";
}

export interface UploadPlan {
  wrangler: string;
  rest: Array<{ method: "POST"; path: string; json?: Record<string, string>; multipart?: { file_name: string } }>;
}

export function uploadPlan(model: string, name: string, folder: string, description = "clef-loop"): UploadPlan {
  // The docs don't state a naming rule; this only keeps the name safe to put in a shell command.
  if (!name || /[\s/'"`$\\]/.test(name)) throw new Error("fine-tune name must be non-empty with no spaces, slashes or quotes");
  return {
    wrangler: `npx wrangler ai finetune create ${model} ${name} ${folder}`,
    rest: [
      { method: "POST", path: "/accounts/{account_id}/ai/finetunes", json: { model, name, description } },
      { method: "POST", path: `/accounts/{account_id}/ai/finetunes/{finetune_id}/finetune-assets`, multipart: { file_name: "adapter_model.safetensors" } },
      { method: "POST", path: `/accounts/{account_id}/ai/finetunes/{finetune_id}/finetune-assets`, multipart: { file_name: "adapter_config.json" } },
    ],
  };
}

export interface UploadDeps {
  accountId: string;
  apiToken: string;
  readFile: (name: string) => Promise<Uint8Array>;
  fetch?: typeof fetch;
  baseUrl?: string;
}

/** Runs the REST half of the plan. Returns the new fine-tune id. */
export async function executeUpload(plan: UploadPlan, deps: UploadDeps): Promise<string> {
  const f = deps.fetch ?? fetch;
  const base = `${deps.baseUrl ?? "https://api.cloudflare.com/client/v4"}/accounts/${encodeURIComponent(deps.accountId)}`;
  const auth = { Authorization: `Bearer ${deps.apiToken}` };
  const [create, ...assets] = plan.rest;
  const res = await f(`${base}/ai/finetunes`, {
    method: "POST",
    headers: { ...auth, "Content-Type": "application/json" },
    body: JSON.stringify(create.json),
  });
  const created = (await res.json()) as { success?: boolean; result?: { id?: string }; errors?: unknown };
  if (!res.ok || !created.success || !created.result?.id) {
    throw new Error(`create fine-tune failed (${res.status}): ${JSON.stringify(created.errors ?? created).slice(0, 200)}`);
  }
  const id = created.result.id;
  for (const step of assets) {
    const fileName = step.multipart!.file_name;
    const form = new FormData();
    form.append("file_name", fileName);
    form.append("file", new Blob([await deps.readFile(fileName)]), fileName);
    const up = await f(`${base}/ai/finetunes/${encodeURIComponent(id)}/finetune-assets`, { method: "POST", headers: auth, body: form });
    const body = (await up.json()) as { success?: boolean; errors?: unknown };
    if (!up.ok || !body.success) throw new Error(`upload ${fileName} failed (${up.status}): ${JSON.stringify(body.errors ?? body).slice(0, 200)}`);
  }
  return id;
}
