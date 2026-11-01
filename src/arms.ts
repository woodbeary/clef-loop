// How to ask each kind of model for a decision, and how to read its answer.
// Runtime-agnostic: the eval CLI sends these over the REST API, the canary Worker
// sends them through the AI binding. Both use the same builders and parsers, so the
// eval measures exactly what production runs.

import { answerFromSystemOne, isObject, optionsFor, validateDecision, type Probabilities } from "./schema.ts";
import type { ArmSpec, Decision, Json, Questions, TemplateName } from "./types.ts";

/** A request to send. `workers-ai` goes to env.AI.run(model, input) or POST /ai/run/{model}. */
export type Call =
  | { transport: "workers-ai"; model: string; input: Record<string, Json> }
  | { transport: "chat"; body: Record<string, Json> }
  | { transport: "url"; url: string; body: Record<string, Json> };

export type CallableArm = Exclude<ArmSpec, { kind: "replay" } | { kind: "majority" }>;

export function buildCall(arm: CallableArm, state: Json, questions: Questions): Call {
  switch (arm.kind) {
    case "systemone":
      return {
        transport: "workers-ai",
        model: arm.model,
        input: { model: arm.selector ?? defaultSelector(arm.model), state, questions: questions as unknown as Json },
      };
    case "systemone-url":
      return {
        transport: "url",
        url: arm.url,
        body: { model: arm.selector ?? "Cloudflare/clef-flash", state, questions: questions as unknown as Json },
      };
    case "lora":
      return {
        transport: "workers-ai",
        model: arm.model,
        input: {
          prompt: renderInference(arm.template, decisionPrompt(state, questions)),
          raw: true,
          max_tokens: maxAnswerTokens(questions),
          temperature: 0,
          ...(arm.lora ? { lora: arm.lora } : {}),
        },
      };
    case "chat": {
      const structured = (arm.format ?? "json_schema") === "json_schema";
      return {
        transport: "chat",
        body: {
          model: arm.model,
          messages: [
            {
              role: "system",
              content: structured
                ? "You make decisions for an operations team. Reply with the JSON object only."
                : "You make decisions for an operations team. Reply with the answer only.",
            },
            { role: "user", content: decisionPrompt(state, questions, structured ? "json" : "text") },
          ],
          ...(structured ? { response_format: decisionResponseFormat(questions) } : {}),
        },
      };
    }
  }
}

/**
 * Chat Completions structured output: a JSON object with one property per question,
 * each limited to that question's option ids, so a frontier model can't lose on format.
 */
export function decisionResponseFormat(questions: Questions): Json {
  const ids = Object.keys(questions);
  return {
    type: "json_schema",
    json_schema: {
      name: "decision",
      strict: true,
      schema: {
        type: "object",
        properties: Object.fromEntries(ids.map((id) => [id, { type: "string", enum: optionsFor(questions[id]) }])),
        required: ids,
        additionalProperties: false,
      },
    },
  };
}

/** Reads a decision from a model's raw result. Throws when it can't, which counts as a miss. */
export function parseResult(arm: CallableArm, raw: unknown, questions: Questions): Decision {
  return parseAnswer(arm, raw, questions).decision;
}

/** Thrown when a model answered but the answer isn't an allowed option. */
export class FormatError extends Error {}

/** The decision, plus per-option probabilities when the arm returns them (System One arms do; text arms don't). */
export function parseAnswer(arm: CallableArm, raw: unknown, questions: Questions): { decision: Decision; probs?: Probabilities } {
  try {
    return parseAnswerUnchecked(arm, raw, questions);
  } catch (e) {
    throw new FormatError((e as Error).message);
  }
}

function parseAnswerUnchecked(arm: CallableArm, raw: unknown, questions: Questions): { decision: Decision; probs?: Probabilities } {
  switch (arm.kind) {
    case "systemone":
    case "systemone-url":
      return answerFromSystemOne(raw, questions);
    case "lora": {
      const text = isObject(raw) && typeof raw.response === "string" ? raw.response : undefined;
      if (text === undefined) throw new Error("text model result has no `response` string");
      return { decision: parseAnswerText(text, questions) };
    }
    case "chat": {
      const choices = isObject(raw) && Array.isArray(raw.choices) ? raw.choices : [];
      const first = choices[0];
      const msg = isObject(first) && isObject(first.message) ? first.message : undefined;
      if (!msg || typeof msg.content !== "string") throw new Error("chat result has no message content");
      return { decision: parseAnswerText(msg.content, questions) };
    }
  }
}

export function defaultSelector(model: string): string {
  const m = model.match(/^@cf\/cloudflare\/(clef(?:-flash|-omni)?)$/);
  return m ? m[1] : model;
}

/** The decision prompt used for LoRA training rows, LoRA inference and chat baselines. */
export function decisionPrompt(state: Json, questions: Questions, answer: "text" | "json" = "text"): string {
  const ids = Object.keys(questions);
  const lines: string[] = [];
  lines.push(
    ids.length === 1 && answer === "text"
      ? "Make the decision below. Answer with the option id only."
      : `Make ${ids.length === 1 ? "the decision" : "each decision"} below. Answer with a JSON object that maps each question id to one option id.`,
  );
  lines.push("", "State:", typeof state === "string" ? state : JSON.stringify(state, null, 2), "", "Questions:");
  for (const id of ids) {
    const q = questions[id];
    lines.push(`- ${id} (${q.type}): ${text(q.instructions)}`);
    const opts = optionsFor(q);
    for (const o of opts) {
      const desc =
        q.type === "choice" ? q.criteria[o] : q.type === "score" ? q.criteria[Number(o)] : q.criteria?.[o as "true" | "false"];
      lines.push(`  ${o}${desc === undefined || desc === null ? "" : `: ${text(desc)}`}`);
    }
  }
  lines.push("", "Answer:");
  return lines.join("\n");
}

/** The answer text a LoRA is trained to produce for a decision. */
export function answerText(decision: Decision, questions: Questions): string {
  const ids = Object.keys(questions);
  return ids.length === 1 ? decision[ids[0]] : JSON.stringify(Object.fromEntries(ids.map((id) => [id, decision[id]])));
}

/**
 * Strict on purpose: the answer must be exactly an option id (allowing quotes,
 * a trailing period, or text on later lines), or a JSON object for multi-question
 * sets. "I would accept this" is a miss, not an accept.
 */
export function parseAnswerText(raw: string, questions: Questions): Decision {
  const ids = Object.keys(questions);
  const body = raw.replace(/^\s*```(?:json)?\s*|\s*```\s*$/g, "").trim();
  if (ids.length === 1) {
    const options = optionsFor(questions[ids[0]]);
    const firstLine = body.split(/\r?\n/)[0].trim().replace(/^["'`]|["'`.]$/g, "").trim();
    const hit = options.find((o) => o.toLowerCase() === firstLine.toLowerCase());
    if (hit) return { [ids[0]]: hit };
  }
  const json = body.match(/\{[\s\S]*\}/);
  if (json) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(json[0]);
    } catch {
      throw new Error(`answer is not valid JSON: ${preview(raw)}`);
    }
    if (isObject(parsed)) {
      const decision = Object.fromEntries(Object.entries(parsed).map(([k, v]) => [k, String(v)]));
      const problems = validateDecision(decision, questions);
      if (problems.length === 0) return decision;
      throw new Error(`answer does not match the questions: ${problems.join("; ")}`);
    }
  }
  throw new Error(`answer is not an option id: ${preview(raw)}`);
}

// Chat templates for the Workers AI LoRA base models named in Cloudflare's AutoTrain
// tutorial (Mistral-7B-Instruct-v0.2, Gemma 2B/7B it, Llama 2 7B chat).
const TEMPLATES: Record<TemplateName, { prefix: string; suffix: string; end: string }> = {
  mistral: { prefix: "<s>[INST] ", suffix: " [/INST]", end: "</s>" },
  llama2: { prefix: "<s>[INST] ", suffix: " [/INST]", end: " </s>" },
  gemma: { prefix: "<bos><start_of_turn>user\n", suffix: "<end_of_turn>\n<start_of_turn>model\n", end: "<end_of_turn>" },
};

export function renderInference(template: TemplateName, prompt: string): string {
  const t = TEMPLATES[template];
  return `${t.prefix}${prompt}${t.suffix}`;
}

export function renderTraining(template: TemplateName, prompt: string, answer: string): string {
  const t = TEMPLATES[template];
  const sep = template === "gemma" ? "" : " ";
  return `${t.prefix}${prompt}${t.suffix}${sep}${answer}${t.end}`;
}

export function isTemplate(s: string): s is TemplateName {
  return s in TEMPLATES;
}

function maxAnswerTokens(questions: Questions): number {
  return Object.keys(questions).length === 1 ? 16 : 32 * Object.keys(questions).length;
}

function text(v: Json | undefined): string {
  return typeof v === "string" ? v : JSON.stringify(v);
}

function preview(s: string): string {
  const one = s.replace(/\s+/g, " ").trim();
  return JSON.stringify(one.length > 60 ? `${one.slice(0, 57)}...` : one);
}
