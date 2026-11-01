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
   `clef/train.jsonl` is the documented request plus a `label` field and who made it.
   That's clef-loop's own format, chosen so the same file can be replayed against any
   System One endpoint. `trainer/train.py` is a supervised trainer for the open weights
   (joint head plus adapters, backbone frozen, cross-entropy plus Brier). Its dry run
   works anywhere. Its full run needs a GPU, and `clef_scorer()` must first be written
   against the release code's `joint_schema_model.py`, which documents inference
   (`load_release_model`, `systemone`) but no differentiable scoring entry point. It has
   never been run. It writes `joint_head.safetensors` and `merged_adapters.safetensors`;
   folding the merged projections into the backbone shards for `byom/` is a manual step.
3. **Reinforcement learning.** Nothing here is RL. RL would mean policy-gradient updates
   driven by the validator's reward with a reference penalty against drift, the way
   Cloudflare describes RLCD; it earns its place only when the validator gives partial
   credit or several answers are acceptable.

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
| Capture from AI Gateway | `clef-loop build --gateway <id> --labels-from <canary URL> --consent <file>` | An API token with AI Gateway Read, the canary's `CLEF_LOOP_TOKEN`, and a consent file you keep current |
| Base and frontier eval arms | `clef-loop eval --arm base=systemone:@cf/cloudflare/clef-flash --arm frontier=chat:openai/gpt-5.5` | An API token with Workers AI Read; Unified Billing credits for `openai/*` |
| LoRA upload | `clef-loop lora upload <dir> ... --execute`, or the printed `wrangler` command | An API token with Workers AI Write |
| Canary Worker | `npx wrangler deploy`, then `npx wrangler secret put CLEF_LOOP_TOKEN` | A Cloudflare account with Workers AI (usage past the free allocation is billed) and SQLite-backed Durable Objects |
| Validator Container | `npx wrangler deploy -c validator/wrangler.jsonc` with Docker running, then the `VALIDATOR` service binding | Containers on the account, and your own check in `validator/rules.ts` |

## Needs a GPU (written, dry run tested, never run)

| Step | Command | Needs |
| --- | --- | --- |
| Tune Clef-flash | `python3 trainer/train.py ... --model-dir <snapshot>` | An H100 or H200, torch 2.11 and transformers 5.10.2, the open weights, and `clef_scorer()` written against `joint_schema_model.py` |
| Serve the tune for eval | SGLang's `/v1/systemone` (model card), then `--arm tuned=systemone-url:<url>` | The same box |

## Verify these on the first real run

1. **`collectLog` per request through the binding.** The logging docs say the
   `cf-aig-collect-log` header overrides the gateway default for one request, either way.
   The Worker uses the binding's equivalent, `collectLog` in `GatewayOptions` (checked in
   `@cloudflare/workers-types` 5.20261014.1). Send one
   `/decide` for a tenant not in `CAPTURE_TENANTS` with the gateway default on, and check
   that no log appears; send one for an opted-in tenant with the default off, and check
   that one does.
2. **Log body shapes.** The API reference types `/logs/{id}/request` and `/response` as
   `unknown`. The exporter accepts the bare System One body, the `/ai/run` envelope, and a
   JSON string inside either. Run `clef-loop audit --gateway <id> ... --limit 5` and check
   that `state` comes out redacted, as the Worker sent it.
3. **Metadata on the list endpoint.** The exporter reads `req`, `tenant` and `route` from
   each listed log's `metadata` and fetches bodies only for consented, labeled requests.
   Check that the list returns the metadata the Worker set.
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
8. **Structured output through the gateway.** The chat arm sends a Chat Completions
   `response_format` of type `json_schema`. Check that `/ai/v1/chat/completions` passes it
   through for the frontier model you compare against; if it doesn't, run that arm with
   `chat:<model>|text` and read the format-failure column.
9. **The validator Container.** `validator/worker.ts` starts the container through
   `ctx.container` and forwards to port 8080, retrying while it boots. The image has not
   been built here. Deploy it with Docker running and call `/health` through the canary's
   service binding before turning `VALIDATE_PERCENT` up.

## Gaps reported as repros

`repros/` turns each platform gap into something Cloudflare can run. `node repros/rest.ts`
prints the REST ones and sends them only with `--execute`; `repros/worker.ts` is a
separate Worker for the two that need a binding. None has been run against Cloudflare.

| Gap | Repro |
| --- | --- |
| No Clef LoRA on Workers AI | `rest.ts clef-lora`: a System One body with a `lora` field |
| Clef-flash's context cut from 64K to 24,576 tokens; the changelog still says 64K | `rest.ts context-24k`: a state of about 30,000 tokens |
| No training code with the open weights | `rest.ts hf-files`: the model's file list on Hugging Face |
| BYOM push isn't public | Nothing to run: no public command or API. `clef-loop byom check` prints the plan |
| `env.AI.aiGatewayLogId` holds only the latest call's id | `worker.ts /log-id-race`: eight concurrent calls, each with its own `req` |
| `patchLog` metadata: merge or replace, undocumented | `worker.ts /patchlog-merge`: patch, then `getLog` |

clef-loop itself no longer depends on the last two: labels are stored by request id and
joined to logs through the `req` metadata, and the Worker never calls `patchLog`. Each call
carries four metadata entries (`req`, `route`, `route_version`, `tenant`), under the
documented five.

## Known limits of clef-loop itself

- **Redaction is pattern-based.** It catches structured PII, sensitive keys, and names
  introduced by a cue. It doesn't catch a free-text name with no cue, or something
  identifying from context alone. The Worker skips capture for a request whose redacted
  state still trips the residual audit, and the build refuses to write when the audit
  finds something, but the audit is also pattern-based. Read a sample with `clef-loop
  audit` before training on real data. Redaction can also hide a field the decision
  needs; mark such keys in `REDACT_KEEP_KEYS` only when they carry no PII.
- **Images can't be pattern-redacted.** A screenshot can show what was typed. Screen-check
  training data must come from test pages you own, never from a user's screen.
- **Withdrawn consent.** Removing a tenant from `CAPTURE_TENANTS` stops new logs; setting
  `until` in the consent file keeps their rows out of every later build; `DELETE /labels`
  removes their labels. The gateway still holds their earlier logs until retention ends
  or you delete them in AI Gateway. clef-loop doesn't delete gateway logs.
- **Consent is configuration.** `CAPTURE_TENANTS` is a Worker variable, so changing it
  means a deploy. Record the consent itself (who, when, what wording) wherever your
  product keeps such records; the consent file only mirrors it for export.
- **Late verdicts.** A validator verdict counts only while its request is still among the
  last 200 of its route (`REVERT_WINDOW`). A verdict that arrives later returns
  `counted: false`.
- **The labels store is one Durable Object per tenant.** That is plenty for a team's
  decisions; a very busy tenant would want labels sharded by day.
- **The eval runs one question set at a time.** Records with different questions are
  rejected, not mixed.
- **Decisions served by a LoRA tuned arm don't come back as training data.** A LoRA
  call logs a rendered prompt, not the System One `state` and `questions`, so `build
  --gateway` skips labeled `route=tuned` logs from a LoRA arm and prints the reason. With a
  10% canary, about 10% of labeled decisions are left out until the tuned arm is a System
  One model (BYOM). Base, fallback and shadow-mode decisions are captured as usual.
- **The canary sees quality only through the validator.** The route reverts on errors,
  timeouts, answers that don't fit the questions, and validator rejects. Without a
  validator bound or posting to `/outcome`, it measures failures only; human labels are
  the slower, fuller comparison (see `MEASURING.md`).
