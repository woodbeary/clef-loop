// Repros for the two binding gaps that need a deployed Worker. Deploy it on its own
// (npx wrangler deploy -c repros/wrangler.jsonc), call each route once, and attach the
// output to the report. clef-loop itself no longer depends on either behavior.
//
//   GET /log-id-race   Fires 8 env.AI.run calls at once, each tagged with its own `req`
//                      metadata, and reads env.AI.aiGatewayLogId after each. Look the ids
//                      up in AI Gateway: any id whose log has another call's `req` is the race.
//   GET /patchlog-merge  Logs one call with metadata {a: 1}, patches {b: 2}, reads the log
//                      back with getLog, and prints its metadata: {a, b} means merge,
//                      {b} means replace. Neither is documented.

interface Env {
  AI: Ai;
  GATEWAY_ID: string;
}

const MODEL = "@cf/cloudflare/clef-flash";
const BODY = { model: "clef-flash", state: "Checkout works.", questions: { ok: { type: "noul", instructions: "Working checkout?" } } };

export default {
  async fetch(request, env): Promise<Response> {
    const path = new URL(request.url).pathname;
    const ai = env.AI as unknown as {
      run(m: string, i: unknown, o: { gateway: { id: string; metadata: Record<string, string | number> } }): Promise<unknown>;
      aiGatewayLogId: string | null;
      gateway(id: string): { patchLog(id: string, d: { metadata: Record<string, number> }): Promise<void>; getLog(id: string): Promise<unknown> };
    };
    if (path === "/log-id-race") {
      const seen = await Promise.all(
        Array.from({ length: 8 }, async (_, i) => {
          await ai.run(MODEL, BODY, { gateway: { id: env.GATEWAY_ID, metadata: { req: `race-${i}` } } });
          return { req: `race-${i}`, aiGatewayLogId: ai.aiGatewayLogId };
        }),
      );
      return Response.json({ note: "look each id up in AI Gateway; a log whose req differs is the race", seen });
    }
    if (path === "/patchlog-merge") {
      await ai.run(MODEL, BODY, { gateway: { id: env.GATEWAY_ID, metadata: { a: 1 } } });
      const id = ai.aiGatewayLogId;
      if (!id) return Response.json({ error: "no log id" }, { status: 500 });
      await ai.gateway(env.GATEWAY_ID).patchLog(id, { metadata: { b: 2 } });
      return Response.json({ id, log: await ai.gateway(env.GATEWAY_ID).getLog(id) });
    }
    return new Response("GET /log-id-race or /patchlog-merge", { status: 404 });
  },
} satisfies ExportedHandler<Env>;
