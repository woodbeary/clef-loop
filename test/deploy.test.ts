import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { describe, it } from "node:test";
import { byomPlan, checkCog } from "../src/byom.ts";
import { checkAdapter, executeUpload, MAX_ADAPTER_BYTES, readSafetensorsHeader, setModelType, uploadPlan } from "../src/lora.ts";

/** Builds the bytes of a minimal safetensors file. */
function safetensors(tensors: Record<string, { dtype: string; shape: number[] }>): Uint8Array {
  let offset = 0;
  const header: Record<string, unknown> = { __metadata__: { format: "pt" } };
  for (const [name, t] of Object.entries(tensors)) {
    const size = t.shape.reduce((a, b) => a * b, 1) * 2;
    header[name] = { ...t, data_offsets: [offset, offset + size] };
    offset += size;
  }
  const json = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + json.length + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  return out;
}

const PEFT_CONFIG = {
  base_model_name_or_path: "mistralai/Mistral-7B-Instruct-v0.2",
  peft_type: "LORA",
  r: 8,
  lora_alpha: 16,
  target_modules: ["q_proj", "v_proj"],
  task_type: "CAUSAL_LM",
};

const ST = safetensors({
  "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": { dtype: "F16", shape: [8, 64] },
  "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": { dtype: "F16", shape: [64, 8] },
});

describe("LoRA adapter check", () => {
  const files = (cfg: object, st = ST, names = ["adapter_config.json", "adapter_model.safetensors"]) => ({
    names,
    configText: JSON.stringify(cfg),
    safetensors: { head: st, size: st.length },
  });

  it("flags the missing model_type that the docs say to add, and passes once it's set", () => {
    const before = checkAdapter(files(PEFT_CONFIG), "@cf/mistral/mistral-7b-instruct-v0.2-lora");
    assert.ok(before.errors.some((e) => /model_type/.test(e)));
    const fixed = JSON.parse(setModelType(JSON.stringify(PEFT_CONFIG), "mistral"));
    const after = checkAdapter(files(fixed), "@cf/mistral/mistral-7b-instruct-v0.2-lora");
    assert.deepEqual(after.errors, []);
    assert.deepEqual(after.info, { rank: 8, modelType: "mistral", baseModel: "mistralai/Mistral-7B-Instruct-v0.2", tensors: 2, dtypes: ["F16"] });
  });

  it("enforces the documented limits", () => {
    const ok = { ...PEFT_CONFIG, model_type: "mistral" };
    assert.ok(checkAdapter(files({ ...ok, r: 64 })).errors.some((e) => /above 32/.test(e)));
    assert.ok(checkAdapter(files({ ...ok, r: 16 })).warnings.some((w) => /r=16/.test(w)));
    assert.ok(checkAdapter(files({ ...ok, model_type: "qwen" })).errors.some((e) => /must be one of/.test(e)));
    assert.ok(checkAdapter(files(ok), "@cf/google/gemma-7b-it-lora").errors.some((e) => /does not match/.test(e)));
    assert.ok(checkAdapter(files(ok, ST, ["adapter_config.json", "adapter.safetensors"])).errors.some((e) => /missing adapter_model.safetensors/.test(e)));
    const big = { ...files(ok), safetensors: { head: ST, size: MAX_ADAPTER_BYTES } };
    assert.ok(checkAdapter(big).errors.some((e) => /300 MB/.test(e)));
    const quant = safetensors({ w: { dtype: "I8", shape: [8, 8] } });
    assert.ok(checkAdapter(files(ok, quant)).errors.some((e) => /quantized/.test(e)));
    assert.throws(() => readSafetensorsHeader(new Uint8Array(4), 4), /too short/);
  });

  it("plans the documented upload and runs it against the REST API", async () => {
    const plan = uploadPlan("@cf/mistral/mistral-7b-instruct-v0.2-lora", "leads-v1", "adapters/leads-v1");
    assert.equal(plan.wrangler, "npx wrangler ai finetune create @cf/mistral/mistral-7b-instruct-v0.2-lora leads-v1 adapters/leads-v1");
    assert.throws(() => uploadPlan("m", "bad name", "d"), /no spaces/);

    const calls: Array<{ url: string; body: unknown }> = [];
    const f = (async (url: string, init: RequestInit) => {
      calls.push({ url, body: init.body });
      if (url.endsWith("/ai/finetunes")) return Response.json({ success: true, result: { id: "ft-123", name: "leads-v1" } });
      return Response.json({ success: true, result: null });
    }) as unknown as typeof fetch;
    const id = await executeUpload(plan, { accountId: "acct", apiToken: "t", fetch: f, readFile: async (n) => new TextEncoder().encode(n) });
    assert.equal(id, "ft-123");
    assert.equal(calls[0].url, "https://api.cloudflare.com/client/v4/accounts/acct/ai/finetunes");
    assert.deepEqual(JSON.parse(String(calls[0].body)), { model: "@cf/mistral/mistral-7b-instruct-v0.2-lora", name: "leads-v1", description: "clef-loop" });
    for (const [i, name] of [[1, "adapter_model.safetensors"], [2, "adapter_config.json"]] as const) {
      assert.equal(calls[i].url, "https://api.cloudflare.com/client/v4/accounts/acct/ai/finetunes/ft-123/finetune-assets");
      const form = calls[i].body as FormData;
      assert.equal(form.get("file_name"), name);
      assert.equal(await (form.get("file") as Blob).text(), name);
    }
  });

  it("stops on an API error", async () => {
    const f = (async () => Response.json({ success: false, errors: [{ message: "nope" }] }, { status: 403 })) as unknown as typeof fetch;
    await assert.rejects(
      executeUpload(uploadPlan("m", "n", "d"), { accountId: "a", apiToken: "t", fetch: f, readFile: async () => new Uint8Array() }),
      /create fine-tune failed \(403\)/,
    );
  });
});

describe("BYOM Cog package", () => {
  it("accepts the package in byom/", async () => {
    const read = (n: string) => readFile(`byom/${n}`, "utf8");
    const res = checkCog({ cogYaml: await read("cog.yaml"), predictPy: await read("predict.py"), requirements: await read("requirements.txt") });
    assert.deepEqual(res, { errors: [], warnings: [] });
  });

  it("reports what's missing", () => {
    assert.deepEqual(checkCog({}).errors, ["missing cog.yaml", "missing predict.py"]);
    const res = checkCog({ cogYaml: "build:\n  python_requirements: requirements.txt\n", predictPy: "class Predictor(BasePredictor):\n  pass\n" });
    assert.ok(res.errors.some((e) => /predict:/.test(e)));
    assert.ok(res.errors.some((e) => /requirements.txt, which is missing/.test(e)));
    assert.ok(res.errors.some((e) => /setup/.test(e)));
    assert.ok(res.warnings.some((w) => /gpu/.test(w)));
  });

  it("says plainly that the push step isn't public", () => {
    assert.ok(byomPlan("t").some((l) => /not public yet/.test(l)));
  });
});
