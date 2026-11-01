# clef-loop

Fine-tune a decision model on your own team's decisions, and ship it on Cloudflare
without betting production on it.

clef-loop is the loop Cloudflare describes for [Clef](https://developers.cloudflare.com/changelog/post/2026-10-01-clef-workers-ai/),
its open decision models on Workers AI, built as a small tool you can read in an
afternoon:

1. **Capture** decisions through the canary Worker, only for tenants who opted in. The
   Worker redacts every state before the model or AI Gateway sees it, and turns gateway
   logging on per request (`collectLog: true`) only for a consenting tenant. Everyone
   else gets a metadata-only event: pTop, margin, latency and a state hash.
2. **Label** with a person's final call or a validator's verdict, never the model's own
   answer. Each label says who decided (`decided_by`: human or validator; agent labels
   are refused), when, and whether the person saw the model's suggestion first.
3. **Split and build datasets.** Export refuses rows without consent. The newest
   decisions are held out. Two training sets are written: Clef System One requests with
   labels, and the `text` CSV that Cloudflare's Workers AI LoRA tutorial specifies.
4. **Evaluate** base Clef, your tuned model and a frontier model (for example
   `openai/gpt-5.5`, asked for structured output) on the same held-out set with the same
   metric. Failures count as wrong, arms are compared pairwise, and Clef's probabilities
   are scored for calibration (Brier, reliability, coverage when low-confidence answers
   go to a person).
5. **Tune** Clef-flash on the open weights: the joint head plus low-rank adapters, with
   the backbone frozen, on cross-entropy plus Brier. That is supervised training; the
   dry run needs no GPU.
6. **Deploy** behind the canary Worker. Per request, it falls back to base when the tuned
   model errors, times out or gives an answer that doesn't fit. A validator in a
   Container (or any caller of `POST /outcome`) reports well-formed wrong answers too, and
   the route reverts when the 95% lower bound of tuned's failure rate passes base's own.

```
 your app ──► canary Worker ──► AI Gateway ──► base Clef  /  tuned model (LoRA, self-hosted, or BYOM)
  (tenant)    redact, consent     (logs only            │
              │  ▲   │            opted-in tenants)     ▼
 staff ──► /feedback │ └─► validator Container ──► verdict ──► revert window (Durable Object)
              │      └──── labels by request id (Durable Object per tenant)
              ▼
 clef-loop build (consent file + labels joined by req) → datasets → eval → trainer → canary
```

TypeScript for everything that touches Cloudflare: the CLI, the API clients, the Worker
and the validator. Python only where the documented path is Python: the Clef-flash trainer
and the Cog package for Bring Your Own Model. The TypeScript has no runtime dependencies,
and the CLI runs directly on Node 22.18+ with no build step.

## What works today, and what needs your account

| Step | Without an account | With a Cloudflare account | Needs gated access |
| --- | --- | --- | --- |
| Capture | CSV export | AI Gateway logs (`--gateway`) | |
| Redact, split, datasets | Yes | Yes | |
| Train | `trainer/train.py --dry-run`: data checks, baseline, and what is trained and frozen | Workers AI LoRA, through Cloudflare's [AutoTrain tutorial](https://developers.cloudflare.com/workers-ai/guides/tutorials/fine-tune-models-with-autotrain/) | Clef-flash itself: `trainer/train.py` on a rented GPU, after its scoring hook is written against the release code |
| Evaluate | `replay` and `majority` baselines; any self-hosted System One server | Base Clef, LoRA arms, frontier models | A BYOM-deployed tuned Clef |
| Deploy | `wrangler deploy --dry-run`; `npm run demo:revert` | Canary Worker with a LoRA as the tuned arm, and the validator Container | Canary with a BYOM tuned Clef |

[docs/GAPS.md](docs/GAPS.md) lists each gated or undocumented step and what to verify on
the first real run. [docs/SOURCES.md](docs/SOURCES.md) cites the Cloudflare page behind
every behavior the code relies on.

## Try it (no account)

```sh
npm ci
npm test              # 128 tests
npm run demo          # build + eval on the synthetic fixtures
npm run demo:revert   # the canary reverting on validator rejects, locally
python3 trainer/train.py --train data/demo/clef/train.jsonl --heldout data/demo/clef/heldout.jsonl --dry-run
```

`npm run demo` runs the first steps on `fixtures/leads.csv`: 303 synthetic home-service
leads with fake names, example.com emails, 555-01XX phone numbers and a test card number.
It writes `data/demo/` and a report like [examples/synthetic-report.md](examples/synthetic-report.md).
The labels come from a rule, so those numbers test the pipeline and say nothing about
any model.

`npm run demo:revert` runs the canary's request handler in Node with the validator's
server on loopback: base answers with the fixtures' rule and "tuned" is a stub that
always gives a well-formed, often wrong answer. The route reverts after 50 tuned
requests on validator rejects alone, and says why.

## Run it on your decisions

```sh
export CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... AI_GATEWAY_ID=clef-loop CLEF_LOOP_TOKEN=...

# 1. Canary first: set CAPTURE_TENANTS to the tenants who opted in, then deploy.
#    Your app sends { key, tenant, state, questions } to /decide, and later
#    { request_id, tenant, final, decided_by, decided_at, saw_suggestion } to /feedback.
npx wrangler secret put CLEF_LOOP_TOKEN
npx wrangler deploy

# 2-3. Export, redact, split, build. consent.json: {"tenants": {"acme": {"from": "2026-10-01T00:00:00Z"}}}
node src/cli.ts audit --gateway clef-loop --labels-from https://<your canary> --consent consent.json --limit 50
node src/cli.ts build --gateway clef-loop --labels-from https://<your canary> --consent consent.json --out data/v1
#    ...or from a CSV with id, created_at, state, final, decided_by, decided_at, saw_suggestion:
node src/cli.ts build --csv decisions.csv --questions questions.json --out data/v1

# 4. Evaluate every arm on the same held-out set.
node src/cli.ts eval --heldout data/v1/clef/heldout.jsonl --train data/v1/clef/train.jsonl \
  --arm logged=replay --arm majority=majority \
  --arm clef=systemone:@cf/cloudflare/clef-flash \
  --arm gpt=chat:openai/gpt-5.5 \
  --abstain-below 0.8 --out reports/v1
# Add --dry-run to print the first request for each arm without sending anything.

# 5. Tune Clef-flash on a GPU box with the open weights (dry run first, anywhere):
python3 trainer/train.py --train data/v1/clef/train.jsonl --heldout data/v1/clef/heldout.jsonl \
  --model-dir weights/clef-flash --dry-run
#    ...or tune a Workers AI LoRA on data/v1/lora/train.csv with the AutoTrain notebook, then:
node src/cli.ts lora check adapters/v1 --model @cf/mistral/mistral-7b-instruct-v0.2-lora --set-model-type mistral
node src/cli.ts lora upload adapters/v1 --model @cf/mistral/mistral-7b-instruct-v0.2-lora --name leads-v1 --execute

# 6. Set TUNED_ARM in wrangler.jsonc, start with MODE=shadow, then canary. Optionally deploy
#    the validator (Docker running) and bind it as VALIDATOR:
npx wrangler deploy -c validator/wrangler.jsonc
```

A tuned Clef served from your own GPU box (the model card documents SGLang's
`/v1/systemone`) can be evaluated today with `--arm tuned=systemone-url:http://host:30000/v1/systemone`.
When BYOM access arrives, `byom/` is the Cog package. `node src/cli.ts byom check` prints
the remaining steps, and the Worker's `TUNED_ARM` becomes
`{"kind":"systemone","model":"<BYOM model id>"}`.

### The Worker's API

All routes need `Authorization: Bearer <CLEF_LOOP_TOKEN>`.

| Route | Body | Does |
| --- | --- | --- |
| `POST /decide` | `{ key?, tenant?, state, questions }` | Redacts the state, picks base or tuned (sticky per `key`), falls back to base on tuned failure, and logs the call in AI Gateway only for a tenant in `CAPTURE_TENANTS`. Returns `{ decision, route, fallback, captured, answers: { q: { p_top, margin } }, needs_person, request_id }`. |
| `POST /feedback` | `{ request_id, tenant, final, decided_by, decided_at, saw_suggestion }` | Stores the label by request id. `decided_by` is `human` or `validator`; `agent` gets 422. Opted-in tenants only. |
| `GET /labels?tenant=` | | Pages through a tenant's labels; `build --gateway` joins them to logs by request id. |
| `DELETE /labels?tenant=` | | Deletes a tenant's labels when consent is withdrawn. |
| `POST /outcome` | `{ request_id, ok, reason? }` | A validator's verdict on what was served. A reject counts as a failure in that route's revert window, once. |
| `GET /route` | | Revert state: both windows, the limit, and whether and why the route reverted. |
| `POST /route/reset` | | Clears a revert after you've fixed the cause. |

The route reverts when the lower end of the 95% Wilson interval for tuned's failure rate
over its last 200 requests is above 1%, or above base's own failure rate when that is
higher, judged once 50 tuned requests are in (`REVERT_*` in `wrangler.jsonc`). One
transient 429 doesn't revert a good canary, and an outage that hits both routes doesn't
blame tuned.

## Design choices

- **The newest decisions are held out**, and exact duplicates are removed first, so no
  input appears on both sides and the eval asks what production asks next.
- **Failures count as wrong.** A timeout, a 429 or an unparseable answer stays in the
  denominator. Answers are parsed strictly: "I would accept this" is a miss, not an
  accept.
- **Nothing reaches a training file unless a tenant opted in, the Worker redacted it
  before the gateway saw it, and a person or a validator supplied the label.** The model
  serves on the same redacted state it is trained on.
- **Labels are joined by request id, not by gateway log id.** `env.AI.aiGatewayLogId`
  holds only the most recent call's id, so the Worker never reads it.
- **Eval traffic is tagged** (`clef_loop_eval` metadata) and skipped by capture, so
  evaluating never contaminates the next training set.
- **One set of request builders and parsers** (`src/arms.ts`) serves both the eval CLI
  and the Worker, so the eval measures what production runs.
- **The canary counts validator rejects as failures.** A well-formed wrong answer is what
  hurts a customer, so the validator's verdicts feed the same window as errors and
  timeouts. Human labels still give the fuller comparison per route over the same days.
  See [docs/MEASURING.md](docs/MEASURING.md).
- **Calibration is measured, not assumed.** Clef returns a probability for every option;
  the eval reports Brier and reliability, and `--abstain-below` (or the Worker's
  `ABSTAIN_BELOW`) sends low-confidence answers to a person.

## What was tested

Run on 2026-11-01 with Node 22.22 and Python 3.13:

- `npm test`: 128 tests, all passing. Beyond v0.1.0's coverage, they drive the Worker's
  capture shim (redaction before the call, `collectLog` per tenant, the residual-PII
  gate, the metadata-only event), labels by request id (agent labels refused, the
  human-over-validator rule, paging, deletion, two calls finishing out of order), the
  Wilson revert rule against base's rate, `/outcome`, the validator's server on loopback
  driving a revert on well-formed wrong answers, consent windows and the label join in
  the gateway exporter, calibration math by hand, structured output for the chat arm,
  the anchoring check, the revert demo and the repro scripts.
- `python -m unittest` in `byom/` (5 tests) and in `trainer/` (11 tests; the torch parity
  test is skipped where torch isn't installed, as it wasn't here).
- `npm run typecheck` for the CLI, the canary Worker, the validator Worker and the repro
  Worker; `wrangler deploy --dry-run` bundles the canary, and the validator Worker with
  `--containers-rollout=none`.
- The validator's server run as its container would run it (`node server.ts`), answering
  `/health` and `/verdict` on loopback.

**Not tested:** no call has been made to Cloudflare or Hugging Face from this repo, no Clef
weights have been loaded, the trainer's full run has never executed (no torch, no GPU,
and its scoring hook needs the release code), the validator's image hasn't been built (no
Docker daemon here), no Cog image has been built, and no LoRA has been trained.

## Layout

```
src/        CLI and library (capture, redact, split, dataset, arms, evaluate, metrics, calibration,
            labels, route-book, lora, byom, canary)
worker/     canary Worker: handler.ts (logic), index.ts (wiring and the Durable Objects)
validator/  the validator: a Node server, its Dockerfile, and the Worker that runs it as a Container
trainer/    supervised Clef-flash trainer (data checks, objective, freeze plan, dry run, tests)
repros/     minimal repros for the platform gaps in docs/GAPS.md (dry run by default)
byom/       Cog package for a tuned Clef (cog.yaml, predict.py, tests)
fixtures/   synthetic decisions, regenerated by scripts/make-fixtures.ts
docs/       SOURCES, GAPS, MEASURING
scripts/    fixture generator and the revert demo
test/       node:test suites
```

## License

MIT. Clef weights are Apache 2.0 from Cloudflare and aren't included here.
