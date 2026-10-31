# Gaps: what isn't public, what needs an account, and what to verify first

clef-loop runs end to end on synthetic data with no account. This page lists every place
where the real loop depends on something that's gated, undocumented or untested, so
nobody mistakes a dry run for a deployment.

## Needs access that isn't public

1. **Pushing a tuned Clef through BYOM.** Cloudflare says BYOM packages models with Cog,
   but the push command, the model id a deployment gets, and how Workers AI maps a request
   onto the predictor's inputs are not public. They were being tested with design
   partners (blog, 2026-04-16). `clef-loop byom check` validates the Cog package and prints
   the plan. It pushes nothing. `byom/predict.py` takes and returns the System One body as
   JSON strings; change its signature to whatever Cloudflare specifies.
2. **Fine-tuning Clef itself.** The launch post offers fine-tuning as a hands-on service
   with Cloudflare's FDE team first, and self-serve later. No upload format is published.
   `clef/train.jsonl` is the documented request plus a `label` field. That's clef-loop's own
   format, chosen so the same file can be replayed against any System One endpoint.
   clef-loop ships no Clef trainer, because the release code's training interface isn't
   documented and no trainer here has been run on a GPU.

## Why the self-serve tune is a LoRA on another base, not on Clef

Workers AI accepts LoRA adapters only for models marked LoRA-capable in its catalog, and
none of the Clef models is one: their model pages list no LoRA support and their input
schemas take no `lora` field. The LoRA guide also caps rank at 32 and lists `model_type`
mistral, gemma or llama, while Clef trains rank-256 adapters on a frozen Qwen backbone
(Qwen3.5-9B for Clef-flash). So "a Clef LoRA on Workers AI" isn't a documented path
today. The `lora` arm tunes a supported base (default
`@cf/mistral/mistral-7b-instruct-v0.2-lora`) as a classifier on the same decisions. It's
the tuned arm you can deploy today, and it's scored against Clef on the same held-out set.

## Needs your Cloudflare account (implemented, tested against fakes, not run against Cloudflare)

| Step | Command | Needs |
| --- | --- | --- |
| Capture from AI Gateway | `clef-loop build --gateway <id>` | An API token with AI Gateway Read |
| Base and frontier eval arms | `clef-loop eval --arm base=systemone:@cf/cloudflare/clef-flash --arm frontier=chat:openai/gpt-5.5` | An API token with Workers AI Read; Unified Billing credits for `openai/*` |
| LoRA upload | `clef-loop lora upload <dir> ... --execute`, or the printed `wrangler` command | An API token with Workers AI Write |
| Canary Worker | `npx wrangler deploy`, then `npx wrangler secret put CLEF_LOOP_TOKEN` | A Cloudflare account with Workers AI (usage past the free allocation is billed) and a SQLite-backed Durable Object |

## Verify these on the first real run

1. **Log body shapes.** The API reference types `/logs/{id}/request` and `/response` as
   `unknown`. The exporter accepts the bare System One body, the `/ai/run` envelope, and a
   JSON string inside either. Run `clef-loop audit --gateway <id> --limit 5` and check that
   `state` comes out as you expect before building a dataset.
2. **`patchLog` metadata: merge or replace.** Not documented. `/feedback` sends the route
   again with the label, so the log works either way. Check one log in the dashboard
   after the first `/feedback`.
3. **`env.AI.aiGatewayLogId` under concurrency.** It's documented to hold only the last
   call's log id. Within one request the Worker never has two calls in flight when it reads
   it (shadow calls start after the base call returns), with one exception: a tuned call
   that timed out keeps running while base serves the fallback, and can finish first. If
   one isolate serves concurrent requests, it could also return another request's id. Each
   call carries a `req` metadata value, so a log can always be found by request id. Compare
   `log_id` with the `req` metadata on a busy route.
4. **The LoRA model id.** The LoRA guide's JavaScript example uses
   `@cf/mistralai/mistral-7b-instruct-v0.2-lora`, while the model page and the REST
   example use `@cf/mistral/mistral-7b-instruct-v0.2-lora`. clef-loop uses the model page's
   id. If it fails, try the other and report the docs mismatch.
5. **`raw: true` and the BOS token.** The `lora` arm sends the full template, including
   `<s>` or `<bos>`. If the server also adds a BOS token, answers can degrade. Compare a few
   outputs with and without the leading token.
6. **Rank wording.** The LoRA guide says rank `r <=8 as well as larger ranks if up to 32`.
   clef-loop accepts up to 32 and warns above 8. AutoTrain's tutorial sets 8.
7. **Log retention.** Accounts whose first gateway was created on or after 2026-09-24
   follow Workers Logs pricing and retention; older accounts keep Legacy Logs. Make sure
   logs outlive your capture window, or run `clef-loop build --gateway` on a schedule and
   keep the redacted output.
8. **Metadata budget.** AI Gateway keeps five custom metadata entries per request.
   `/decide` uses three (`route`, `route_version`, `req`) and `/feedback` uses three
   (`final`, `route`, `route_version`). If `patchLog` merges (item 2), a labeled log holds
   four, so add at most one of your own; if it replaces, two.

## Known limits of clef-loop itself

- **Redaction is pattern-based.** It catches structured PII and names you list. It
  doesn't catch a free-text name with no cue, or something identifying from context
  alone. The build refuses to write when its residual audit finds something, but the
  audit is also pattern-based. Read a sample with `clef-loop audit` before training on
  real data.
- **The eval runs one question set at a time.** Records with different questions are
  rejected, not mixed.
- **Decisions served by a LoRA tuned arm don't come back as training data.** A LoRA
  call logs a rendered prompt, not the System One `state` and `questions`, so `build
  --gateway` skips labeled `route=tuned` logs from a LoRA arm and prints the reason. With a
  10% canary, about 10% of labeled decisions are left out until the tuned arm is a System
  One model (BYOM). Base, fallback and shadow-mode decisions are captured as usual.
- **The canary measures failures, not quality.** The route reverts on errors, timeouts
  and answers that don't fit the questions. Whether the tuned model decides better shows
  up later, in the human labels attached through `/feedback`. Compare arms there (see
  `MEASURING.md`).
