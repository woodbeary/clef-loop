// PII redaction for decision records, applied before anything is written to disk
// or used for training.
//
// What it catches: emails, phone numbers, payment card numbers (Luhn-checked),
// US SSNs, IPv4 addresses, URLs, street addresses with a number and a street
// suffix, values under sensitive keys (name, email, phone, address, ...), names you
// list (whole, and each capitalized part of 3+ letters), names introduced as
// "my name is X" / "this is X", and sign-offs like "Thanks, X".
//
// What it does not catch: a free-text name with none of those cues, a street address
// without a number and suffix, or anything identifying in context alone. Run
// `clef-loop audit` and read a sample before you train on real data.

import type { Json } from "./types.ts";

export interface RedactOptions {
  /** Known names (for example, from your CRM) to remove wherever they appear. Each part
   * of 3+ letters is also removed where it appears capitalized ("Dana" from "Dana Example"). */
  names?: string[];
  /** Keys that look sensitive but carry no PII in your data, kept as-is. */
  keepKeys?: string[];
}

export type RedactionCounts = Record<string, number>;

const PATTERNS: Array<[label: string, re: RegExp]> = [
  ["URL", /\bhttps?:\/\/[^\s<>"')]+/gi],
  ["EMAIL", /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g],
  // Card before phone: 13-19 digits, optionally grouped. Checked with Luhn below.
  ["CARD", /(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])/g],
  ["SSN", /(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)/g],
  ["PHONE", /(?<![\d+])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)/g],
  ["IP", /(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])/g],
  [
    "ADDRESS",
    /\b\d{1,6}\s+(?:[A-Za-z0-9.'-]+\s+){1,4}?(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl|Parkway|Pkwy|Highway|Hwy|Circle|Cir|Terrace|Ter)\b\.?(?:,?\s*(?:#|Apt\.?|Unit|Suite|Ste\.?)\s*[A-Za-z0-9-]+)?/gi,
  ],
];

const INTRODUCED_NAME = /\b(my name is|this is|name's)\s+([A-Z][a-z]+(?:[ -][A-Z][a-z]+)?)/g;
const SIGN_OFF = /\b([Tt]hanks|[Tt]hank you|[Rr]egards|[Bb]est|[Cc]heers|[Ss]incerely)(,?\s+|,\s*\n\s*)(?:-\s*)?([A-Z][a-z]+)(?=\s*[.!]?\s*$)/gm;

const KEY_LABELS: Array<[tokens: string[], label: string]> = [
  [["email", "mail"], "EMAIL"],
  [["phone", "mobile", "cell", "tel", "telephone"], "PHONE"],
  [["address", "street", "addr"], "ADDRESS"],
  [["ssn"], "SSN"],
  [["ip"], "IP"],
  [["card", "cc"], "CARD"],
  [["dob", "birthday", "birthdate"], "DOB"],
  [["name", "firstname", "lastname", "fullname"], "NAME"],
];

export function redactText(text: string, opts: RedactOptions = {}, counts: RedactionCounts = {}): string {
  let out = text;
  for (const [label, re] of PATTERNS) {
    out = out.replace(re, (m) => {
      if (label === "CARD" && !luhn(m.replace(/\D/g, ""))) return m;
      bump(counts, label);
      return `[${label}]`;
    });
  }
  out = out.replace(INTRODUCED_NAME, (_m, cue: string) => {
    bump(counts, "NAME");
    return `${cue} [NAME]`;
  });
  out = out.replace(SIGN_OFF, (_m, cue: string, sep: string) => {
    bump(counts, "NAME");
    return `${cue}${sep}[NAME]`;
  });
  const names = (opts.names ?? []).map((n) => n.trim()).filter((n) => n.length >= 2);
  // Full names first, case-insensitive; then capitalized parts, case-sensitive.
  for (const name of names) out = replaceWord(out, name, "giu", counts);
  const parts = new Set(names.flatMap((n) => n.split(/\s+/)).filter((p) => p.length >= 3 && /^\p{Lu}/u.test(p)));
  for (const part of parts) out = replaceWord(out, part, "gu", counts);
  return out;
}

/** Redacts every string in a JSON value. Values under sensitive keys are replaced whole. */
export function redactJson(value: Json, opts: RedactOptions = {}, counts: RedactionCounts = {}): Json {
  if (typeof value === "string") return redactText(value, opts, counts);
  if (Array.isArray(value)) return value.map((v) => redactJson(v, opts, counts));
  if (value === null || typeof value !== "object") return value;
  const keep = new Set((opts.keepKeys ?? []).map((k) => k.toLowerCase()));
  const out: Record<string, Json> = {};
  for (const [k, v] of Object.entries(value)) {
    const label = keep.has(k.toLowerCase()) ? undefined : sensitiveKeyLabel(k);
    if (label && (typeof v === "number" || (typeof v === "string" && v !== ""))) {
      bump(counts, label);
      out[k] = `[${label}]`;
    } else {
      out[k] = redactJson(v, opts, counts);
    }
  }
  return out;
}

export interface ResidualFinding {
  path: string;
  kind: "email-like" | "long-digit-run" | "at-handle";
  sample: string;
}

/**
 * Scans already-redacted data for things that still look like PII. The dataset
 * builder refuses to write when this finds anything, unless told otherwise.
 */
export function auditResidual(value: Json, path = "$"): ResidualFinding[] {
  const findings: ResidualFinding[] = [];
  if (typeof value === "string") {
    const checks: Array<[ResidualFinding["kind"], RegExp]> = [
      ["email-like", /\S+@\S+\.\S+/],
      ["long-digit-run", /\d(?:[\s().-]?\d){6,}/],
      ["at-handle", /(?<![\w.])@[A-Za-z0-9_]{3,}/],
    ];
    for (const [kind, re] of checks) {
      const m = value.match(re);
      if (m) findings.push({ path, kind, sample: mask(m[0]) });
    }
  } else if (Array.isArray(value)) {
    value.forEach((v, i) => findings.push(...auditResidual(v, `${path}[${i}]`)));
  } else if (value !== null && typeof value === "object") {
    for (const [k, v] of Object.entries(value)) findings.push(...auditResidual(v, `${path}.${k}`));
  }
  return findings;
}

export function sensitiveKeyLabel(key: string): string | undefined {
  const tokens = key
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
  const joined = tokens.join("");
  for (const [names, label] of KEY_LABELS) {
    if (tokens.some((t) => names.includes(t)) || names.includes(joined)) return label;
  }
  return undefined;
}

export function luhn(digits: string): boolean {
  if (digits.length < 13 || digits.length > 19) return false;
  let sum = 0;
  for (let i = 0; i < digits.length; i++) {
    let d = Number(digits[digits.length - 1 - i]);
    if (i % 2 === 1) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
  }
  return sum % 10 === 0;
}

function replaceWord(text: string, word: string, flags: string, counts: RedactionCounts): string {
  const re = new RegExp(`(?<![\\p{L}\\p{N}])${escapeRegExp(word)}(?![\\p{L}\\p{N}])`, flags);
  return text.replace(re, () => {
    bump(counts, "NAME");
    return "[NAME]";
  });
}

function bump(counts: RedactionCounts, label: string): void {
  counts[label] = (counts[label] ?? 0) + 1;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function mask(s: string): string {
  return s.length <= 4 ? "****" : `${s.slice(0, 2)}${"*".repeat(Math.min(s.length - 4, 12))}${s.slice(-2)}`;
}
