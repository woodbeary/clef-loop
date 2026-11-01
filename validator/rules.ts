// The example validator's rule, for the synthetic lead decisions in fixtures/. It
// decides from the facts of the case, never from the model's answer, which it is never
// sent. Replace verdict() with your own check: for a screen check, the state your test
// page knows it rendered; for an agent action, whether the next step succeeded.
// Return null when the check can't tell, and no verdict is recorded. The canary Worker
// also ignores a verdict that isn't an allowed option for every question.

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export type Decision = Record<string, string>;

export function verdict(state: Json, questions: Record<string, unknown>): Decision | null {
  if (!("decision" in questions) || typeof state !== "object" || state === null || Array.isArray(state)) return null;
  const s = state as Record<string, Json>;
  if (typeof s.distance_miles !== "number" || typeof s.hazardous_items_mentioned !== "boolean") return null;
  if (s.hazardous_items_mentioned) return { decision: "skip" };
  if (s.distance_miles > 30) return { decision: "skip" };
  if (s.estimated_load === "unknown" && s.photos_attached !== true) return { decision: "needs_info" };
  if (s.estimated_load === "single item" && s.distance_miles > 15) return { decision: "skip" };
  return { decision: "accept" };
}
