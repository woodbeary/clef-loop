// The validator's Worker: a Durable Object that owns the container (validator/Dockerfile)
// and forwards each request to it. The canary Worker reaches it through a service
// binding named VALIDATOR (see wrangler.jsonc in the repository root).

import { DurableObject } from "cloudflare:workers";

interface Env {
  VALIDATOR_CONTAINER: DurableObjectNamespace<ValidatorContainer>;
}

const PORT = 8080;

export default {
  async fetch(request, env): Promise<Response> {
    return env.VALIDATOR_CONTAINER.get(env.VALIDATOR_CONTAINER.idFromName("validator")).fetch(request);
  },
} satisfies ExportedHandler<Env>;

export class ValidatorContainer extends DurableObject<Env> {
  async fetch(request: Request): Promise<Response> {
    const container = this.ctx.container;
    if (!container) return new Response("this Durable Object has no container", { status: 500 });
    if (!container.running) container.start({ enableInternet: false });
    const url = new URL(request.url);
    const target = `http://container:${PORT}${url.pathname}${url.search}`;
    for (let attempt = 0; ; attempt++) {
      try {
        return await container.getTcpPort(PORT).fetch(new Request(target, request.clone()));
      } catch (e) {
        if (attempt >= 20) throw e;
        await new Promise((r) => setTimeout(r, 250));
      }
    }
  }
}
