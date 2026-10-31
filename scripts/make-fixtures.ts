// Writes fixtures/ from scripts/synthetic.ts. Deterministic: running it twice gives
// identical files, and a test checks the committed files match.

import { mkdir, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { toCsv } from "../src/csv.ts";
import { FAKE_NAMES, QUESTIONS, syntheticLeads } from "./synthetic.ts";

export function fixtureFiles(): Record<string, string> {
  const leads = syntheticLeads();
  const csv = toCsv(
    ["id", "created_at", "state", "model_output", "final"],
    leads.map((l) => [l.id, l.createdAt, JSON.stringify(l.state), l.modelOutput!.decision, l.final.decision]),
  );
  const sample = {
    model: "clef-flash",
    state: { service: "garage cleanout", estimated_load: "half truck", distance_miles: 8.5, photos_attached: true },
    questions: QUESTIONS,
  };
  return {
    "fixtures/questions.json": JSON.stringify(QUESTIONS, null, 2) + "\n",
    "fixtures/leads.csv": csv,
    "fixtures/names.txt": FAKE_NAMES.join("\n") + "\n",
    "fixtures/systemone-request.json": JSON.stringify(sample, null, 2) + "\n",
  };
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  await mkdir("fixtures", { recursive: true });
  for (const [path, text] of Object.entries(fixtureFiles())) await writeFile(path, text);
  console.log("wrote fixtures/");
}
