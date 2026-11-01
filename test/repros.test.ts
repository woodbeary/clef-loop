import assert from "node:assert/strict";
import { describe as suite, it } from "node:test";
import { describe, execute, repros } from "../repros/rest.ts";

suite("repros", () => {
  it("dry run prints each request without an account id and sends nothing", () => {
    const all = repros();
    assert.deepEqual(all.map((r) => r.name), ["clef-lora", "context-24k", "hf-files"]);
    for (const r of all) {
      const text = describe(r);
      assert.match(text, /^# /);
      assert.match(text, /expect: /);
      if (r.needsAccount) assert.match(text, /\/accounts\/\{account\}\//);
    }
    const lora = all[0].body as Record<string, unknown>;
    assert.equal(lora.lora, "clef-loop-repro");
    const long = all[1].body as { state: string };
    assert.ok(long.state.length > 200_000);
  });

  it("executes only with credentials, and fills them in", async () => {
    const [lora, , hf] = repros();
    await assert.rejects(execute(lora, {}), /needs CLOUDFLARE_ACCOUNT_ID/);
    const calls: Array<{ url: string; init: RequestInit }> = [];
    const f = (async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return new Response('{"success":false}', { status: 400 });
    }) as unknown as typeof fetch;
    assert.equal(await execute(lora, { CLOUDFLARE_ACCOUNT_ID: "acct", CLOUDFLARE_API_TOKEN: "tok" }, f), 'clef-lora: 400 {"success":false}');
    assert.equal(calls[0].url, "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef-flash");
    assert.equal((calls[0].init.headers as Record<string, string>).Authorization, "Bearer tok");
    await execute(hf, {}, f);
    assert.equal((calls[1].init.headers as Record<string, string>).Authorization, undefined);
  });
});
