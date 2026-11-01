// Shared types. Everything here is runtime-agnostic: it runs in Node (the CLI)
// and in a Worker (the canary).
//
// The question and answer shapes follow the Clef / System One API as published on
// developers.cloudflare.com/workers-ai/models/clef-flash/ (schema-input.json and
// schema-output.json, checked 2026-10-31).

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

export interface NoulQuestion {
  type: "noul";
  instructions: Json;
  criteria?: { true?: Json; false?: Json };
}

export interface ChoiceQuestion {
  type: "choice";
  instructions: Json;
  /** Option id -> description. 2 to 255 options. */
  criteria: Record<string, Json>;
}

export interface ScoreQuestion {
  type: "score";
  instructions: Json;
  /** Ordered levels, lowest first, indexed from 0. 2 to 10 levels. */
  criteria: Json[];
}

export type Question = NoulQuestion | ChoiceQuestion | ScoreQuestion;
export type Questions = Record<string, Question>;

/**
 * One decision, normalized to a single option id per question:
 * - choice: the option id ("accept")
 * - noul:   "true" or "false"
 * - score:  the level index as a string ("0".."9")
 */
export type Decision = Record<string, string>;

/** A labeled decision: what the model saw, what it said, and what the human decided. */
export interface DecisionRecord {
  id: string;
  /** ISO 8601 time the decision was made. Used for the time-based split. */
  createdAt: string;
  state: Json;
  questions: Questions;
  /** The answer the production model gave at the time, if it was logged. */
  modelOutput?: Decision;
  /** The final decision: a person's call or a validator's verdict, never the model's own answer. This is the label. */
  final: Decision;
  /** Who made the final decision. Agent-made labels are refused at capture. */
  decidedBy: "human" | "validator";
  /** ISO 8601 time the final decision was made. */
  decidedAt: string;
  /** Whether the person saw the model's suggestion before deciding. Always false for a validator. */
  sawSuggestion: boolean;
  /** The tenant whose consent covers this record, when it came through the canary Worker. */
  tenant?: string;
  /** Where the record came from, for the manifest. */
  source: "csv" | "ai-gateway" | "dataset";
}

/** How to call one model. Shared by the eval CLI and the canary Worker. */
export type ArmSpec =
  | {
      /** A System One model on Workers AI: Clef, Clef-flash, Clef-omni, or a BYOM deployment. */
      kind: "systemone";
      model: string;
      /** The body's `model` field. Clef models require "clef", "clef-flash" or "clef-omni" (one per model). */
      selector?: string;
    }
  | {
      /** A System One server you host yourself, for example SGLang's /v1/systemone. */
      kind: "systemone-url";
      url: string;
      selector?: string;
    }
  | {
      /** A Workers AI text model that accepts a LoRA adapter (raw prompt + `lora`). */
      kind: "lora";
      model: string;
      /** Fine-tune name or id. Omit to run the base model with the same prompt. */
      lora?: string;
      template: TemplateName;
    }
  | {
      /** A chat model through AI Gateway, for example openai/gpt-5.5. */
      kind: "chat";
      model: string;
      /** "json_schema" (default) asks for structured output limited to the allowed options; "text" asks for the bare answer. */
      format?: "json_schema" | "text";
    }
  | {
      /** The answer the production model logged at capture time. No call is made. */
      kind: "replay";
    }
  | {
      /** Always predicts the most common training label. No call is made. */
      kind: "majority";
    };

export type TemplateName = "mistral" | "gemma" | "llama2";

/** The result of running one arm on one record. */
export type ArmOutcome =
  | { ok: true; decision: Decision; ms: number; probs?: Record<string, Record<string, number>> }
  | { ok: false; error: string; ms: number; kind?: FailureKind };

/** `call`: the request failed (network, status, timeout). `format`: the model answered, but not with an allowed option. */
export type FailureKind = "call" | "format";
