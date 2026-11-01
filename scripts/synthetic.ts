// Synthetic lead decisions for tests and the demo. Every name, email, phone number,
// address and card number here is fake: emails use example.com (RFC 2606), phone
// numbers use the 555-01XX fiction range, and the card is a public test number.
//
// The "human" labels follow a fixed rule plus 4% noise, each with a decision time a few
// minutes to two hours later and whether the person saw the model's suggestion first. The "logged model output"
// follows a weaker rule, standing in for what a base model said at the time. Neither
// is a real model; numbers computed on this data test the pipeline, nothing more.

import type { Decision, DecisionRecord, Questions } from "../src/types.ts";

export const QUESTIONS: Questions = {
  decision: {
    type: "choice",
    instructions: "Should we take this lead?",
    criteria: {
      accept: "Book it: inside our service area, a job we do, and enough detail to quote.",
      skip: "Decline: too far, hazardous materials, or not worth the trip.",
      needs_info: "Ask for photos or details before deciding.",
    },
  },
};

export const FAKE_NAMES = ["Dana Example", "Riley Sample", "Jordan Placeholder", "Casey Testcase", "Morgan Fixture", "Avery Mock"];

const SERVICES = ["furniture pickup", "garage cleanout", "appliance removal", "yard debris", "estate cleanout", "construction debris", "mattress pickup"];
const LOADS = ["single item", "quarter truck", "half truck", "full truck", "unknown"];
const WINDOWS = ["today", "this week", "next week", "flexible"];
const CHANNELS = ["web_form", "sms", "voicemail_transcript"];

/** Small deterministic PRNG (mulberry32), so fixtures are identical on every run. */
export function prng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Lead {
  channel: string;
  service: string;
  estimated_load: string;
  distance_miles: number;
  requested_window: string;
  photos_attached: boolean;
  hazardous_items_mentioned: boolean;
  customer_name: string;
  phone: string;
  message: string;
}

export function humanRule(l: Lead): string {
  if (l.hazardous_items_mentioned) return "skip";
  if (l.distance_miles > 30) return "skip";
  if (l.estimated_load === "unknown" && !l.photos_attached) return "needs_info";
  if (l.estimated_load === "single item" && l.distance_miles > 15) return "skip";
  return "accept";
}

/** A weaker rule standing in for a base model: misses hazards half the time, uses a
 * looser distance limit, and guesses on unknown loads. */
function loggedRule(l: Lead, r: () => number): string {
  if (l.hazardous_items_mentioned && r() < 0.5) return "skip";
  if (l.distance_miles > 40) return "skip";
  if (l.estimated_load === "unknown" && !l.photos_attached) return r() < 0.5 ? "accept" : "needs_info";
  return "accept";
}

export function syntheticLeads(n = 300, seed = 20261031): DecisionRecord[] {
  const r = prng(seed);
  const who = prng(seed + 1);
  const pick = <T>(xs: T[]): T => xs[Math.floor(r() * xs.length)];
  const out: DecisionRecord[] = [];
  let t = Date.parse("2026-08-01T15:00:00Z");
  for (let i = 1; i <= n; i++) {
    t += Math.floor((3 + r() * 5) * 3600_000);
    const name = pick(FAKE_NAMES);
    const first = name.split(" ")[0];
    const phone = `555-555-01${String(Math.floor(r() * 100)).padStart(2, "0")}`;
    const service = pick(SERVICES);
    const window = pick(WINDOWS);
    const hazardous = r() < 0.08;
    const photos = r() < 0.45;
    const messages = [
      `Hi, this is ${name}. Need ${service} at ${100 + Math.floor(r() * 900)} Example St. Call me at ${phone} or ${first.toLowerCase()}@example.com.`,
      `my name is ${name}, can you come ${window}? ${photos ? "Sent photos." : "No photos yet."}`,
      `Need help asap with ${service}. Text ${phone}.`,
      `${service}, ${window}. Can I pay with card 4111 1111 1111 1111?`,
      `${hazardous ? "There are old paint cans and some chemicals too. " : ""}Mostly ${service}. Thanks, ${first}`,
    ];
    const lead: Lead = {
      channel: pick(CHANNELS),
      service,
      estimated_load: pick(LOADS),
      distance_miles: Math.round(r() * 450) / 10,
      requested_window: window,
      photos_attached: photos,
      hazardous_items_mentioned: hazardous,
      customer_name: name,
      phone,
      message: pick(messages),
    };
    let label = humanRule(lead);
    if (r() < 0.04) label = pick(["accept", "skip", "needs_info"].filter((x) => x !== label));
    const final: Decision = { decision: label };
    const modelOutput: Decision = { decision: loggedRule(lead, r) };
    out.push({
      id: `lead-${String(i).padStart(4, "0")}`,
      createdAt: new Date(t).toISOString(),
      state: lead as unknown as DecisionRecord["state"],
      questions: QUESTIONS,
      modelOutput,
      final,
      decidedBy: "human",
      decidedAt: new Date(t + Math.floor((5 + who() * 115) * 60_000)).toISOString(),
      sawSuggestion: who() < 0.5,
      source: "csv",
    });
  }
  // A few exact resubmissions, so dedupe has something to do.
  for (const k of [10, 50, 90]) {
    const src = out[k];
    t += 3600_000;
    out.push({ ...src, id: `${src.id}-dup`, createdAt: new Date(t).toISOString(), decidedAt: new Date(t + 600_000).toISOString() });
  }
  return out;
}
