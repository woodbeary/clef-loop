// The validator that runs in a Cloudflare Container (validator/Dockerfile). Plain Node,
// no dependencies:
//
//   POST /verdict { request_id, state, questions } -> { request_id, final } or { request_id, final: null, reason }
//   GET  /health                                   -> { ok: true }
//
// It receives the redacted state and the questions, never the model's answer, so it
// can't be gamed by agreement. The canary Worker compares `final` with what it served.

import { createServer, type Server } from "node:http";
import { pathToFileURL } from "node:url";
import { verdict, type Decision } from "./rules.ts";

export type Rule = (state: unknown, questions: Record<string, unknown>) => Decision | null;

export function createValidatorServer(rule: Rule = verdict as Rule): Server {
  return createServer((req, res) => {
    const send = (status: number, body: unknown) => {
      res.writeHead(status, { "Content-Type": "application/json" });
      res.end(JSON.stringify(body));
    };
    if (req.method === "GET" && req.url === "/health") return send(200, { ok: true });
    if (req.method !== "POST" || req.url !== "/verdict") return send(404, { error: "not found" });
    let raw = "";
    req.setEncoding("utf8");
    req.on("data", (c: string) => {
      raw += c;
      if (raw.length > 1_000_000) req.destroy();
    });
    req.on("end", () => {
      let body: { request_id?: unknown; state?: unknown; questions?: unknown };
      try {
        body = JSON.parse(raw);
      } catch {
        return send(400, { error: "body must be JSON" });
      }
      if (typeof body.request_id !== "string" || typeof body.questions !== "object" || body.questions === null) {
        return send(400, { error: "request_id and questions are required" });
      }
      const final = rule(body.state, body.questions as Record<string, unknown>);
      send(200, final ? { request_id: body.request_id, final } : { request_id: body.request_id, final: null, reason: "the check can't tell" });
    });
  });
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const port = Number(process.env.PORT ?? "8080");
  createValidatorServer().listen(port, "0.0.0.0", () => console.log(`validator listening on ${port}`));
}
