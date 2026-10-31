# clef-loop

Fine-tune a decision model on your own team's decisions, and ship it on Cloudflare
without betting production on it.

clef-loop is the loop Cloudflare describes for [Clef](https://developers.cloudflare.com/changelog/post/2026-10-01-clef-workers-ai/),
its open decision models on Workers AI, built as a small tool you can read in an
afternoon:

1. **Capture** labeled decisions: what the model saw, what it answered, and what the
   human finally decided. Input is AI Gateway logs or a CSV export.
2. **Redact** PII before anything is written, then audit what's left. The build refuses
   to write if anything still looks like PII.
3. **Split and build datasets.** The newest decisions are held out. Two training sets are
   written: Clef System One requests with labels, and the `text` CSV that Cloudflare's
   Workers AI LoRA tutorial specifies.
4. **Evaluate** base Clef, your tuned model and a frontier model (for example
   `openai/gpt-5.5`) on the same held-out set with the same metric. Failures count as
   wrong, and arms are compared pairwise.
5. **Deploy** the tuned model behind a canary Worker. Per request, it falls back to base
   when the tuned model errors, times out or gives an answer that doesn't fit. When
   failures pass a threshold, the whole route reverts automatically. Every call is logged
   through AI Gateway, so production builds the next training set.

```
 your app ──► canary Worker ──► AI Gateway ──► base Clef  /  tuned model (LoRA or BYOM)
                 │    ▲                │
   staff decide ─┘    └ revert window  └─► logs + /feedback labels
                      (Durable Object)        │
                                              ▼
            clef-loop build: capture → redact → split → datasets → eval → tune → deploy
```

TypeScript for everything that touches Cloudflare: the CLI, the API clients and the
Worker. Python only where the documented path is Python: the Cog package for Bring Your
Own Model. The TypeScript has no runtime dependencies, and the CLI runs directly on Node
22.18+ with no build step.

## What works today, and what needs your account

| Step | Without an account | With a Cloudflare account | Needs gated access |
| --- | --- | --- | --- |
| Capture | CSV export | AI Gateway logs (`--gateway`) | |
| Redact, split, datasets | Yes | Yes | |
| Train | | Workers AI LoRA, through Cloudflare's [AutoTrain tutorial](https://developers.cloudflare.com/workers-ai/guides/tutorials/fine-tune-models-with-autotrain/) | Clef itself, through Cloudflare's fine-tuning service |
| Evaluate | `replay` and `majority` baselines; any self-hosted System One server | Base Clef, LoRA arms, frontier models | A BYOM-deployed tuned Clef |
| Deploy | `wrangler deploy --dry-run` | Canary Worker with a LoRA as the tuned arm | Canary with a BYOM tuned Clef |

[docs/GAPS.md](docs/GAPS.md) lists each gated or undocumented step and what to verify on
the first real run. [docs/SOURCES.md](docs/SOURCES.md) cites the Cloudflare page behind
every behavior the code relies on.

## Try it (no account)

```sh
npm ci
npm test          # 84 tests
npm run demo      # build + eval on the synthetic fixtures
```

`npm run demo` runs the first steps on `fixtures/leads.csv`: 303 synthetic home-service
leads with fake names, example.com emails, 555-01XX phone numbers and a test card number.
It writes `data/demo/` and a report like [examples/synthetic-report.md](examples/synthetic-report.md).
The labels come from a rule, so those numbers test the pipeline and say nothing about
any model.

## Run it on your decisions

```sh
export CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... AI_GATEWAY_ID=clef-loop

# 1-3. Capture, redact, split, build. From AI Gateway logs labeled through /feedback:
node src/cli.ts audit --gateway clef-loop --limit 50          # read a redacted sample first
node src/cli.ts build --gateway clef-loop --out data/v1 --names crm-names.txt
#    ...or from a CSV with id, created_at, state, final, model_output:
node src/cli.ts build --csv decisions.csv --questions questions.json --out data/v1

# Train a LoRA on data/v1/lora/train.csv with the AutoTrain notebook (lora-r 8, no
# quantization), download adapter_config.json + adapter_model.safetensors, then:
node src/cli.ts lora check adapters/v1 --model @cf/mistral/mistral-7b-instruct-v0.2-lora --set-model-type mistral
node src/cli.ts lora upload adapters/v1 --model @cf/mistral/mistral-7b-instruct-v0.2-lora --name leads-v1 --execute

# 4. Evaluate every arm on the same held-out set.
node src/cli.ts eval --heldout data/v1/clef/heldout.jsonl --train data/v1/clef/train.jsonl \
  --arm logged=replay --arm majority=majority \
  --arm clef=systemone:@cf/cloudflare/clef-flash \
  --arm tuned='lora:@cf/mistral/mistral-7b-instruct-v0.2-lora|leads-v1|mistral' \
  --arm gpt=chat:openai/gpt-5.5 \
  --out reports/v1
# Add --dry-run to print the first request for each arm without sending anything.

# 5. Canary. Set TUNED_ARM in wrangler.jsonc, start with MODE=shadow, then canary.
npx wrangler secret put CLEF_LOOP_TOKEN
npx wrangler deploy
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
| `POST /decide` | `{ key?, state, questions }` | Picks base or tuned (sticky per `key`). Falls back to base on tuned failure. Returns `{ decision, route, fallback, log_id, request_id }`. |
| `POST /feedback` | `{ log_id, final, route }` | Attaches the human's final decision to the AI Gateway log. This is what `build --gateway` reads. |
| `GET /route` | | Revert state: failures in the window, and whether and why the route reverted. |
| `POST /route/reset` | | Clears a revert after you've fixed the cause. |

By default the route reverts when more than 1% of the last 200 tuned requests fell back,
judged once 50 are in the window (`REVERT_*` in `wrangler.jsonc`).

## Design choices

- **The newest decisions are held out**, and exact duplicates are removed first, so no
  input appears on both sides and the eval asks what production asks next.
- **Failures count as wrong.** A timeout, a 429 or an unparseable answer stays in the
  denominator. Answers are parsed strictly: "I would accept this" is a miss, not an
  accept.
- **Eval traffic is tagged** (`clef_loop_eval` metadata) and skipped by capture, so
  evaluating never contaminates the next training set.
- **One set of request builders and parsers** (`src/arms.ts`) serves both the eval CLI
  and the Worker, so the eval measures what production runs.
- **The canary watches failures, not quality.** Quality shows up in the human labels,
  which you compare per route over the same days. See [docs/MEASURING.md](docs/MEASURING.md).

## What was tested

Run on 2026-10-31 with Node 22.22 and Python 3.13:

- `npm test`: 84 tests, all passing. They cover the documented Clef request and response
  shapes; every redaction pattern and the residual audit; the CSV parser; the time split
  and dedupe; dataset files (no PII left in any of them, AutoTrain CSV format, counts that
  add up); the AI Gateway exporter against a fake of the logs API (paging, eval-tag
  skipping, both body shapes); every arm's request and parser; metrics checked against
  hand-computed values (precision and recall, Wilson interval, exact McNemar); the REST
  transport (endpoints, headers, retries); one eval over real HTTP against a local
  System One server; the LoRA adapter checks and upload calls; the canary rules; the
  Worker's handler driven end to end with a fake AI binding (canary, fallback on error,
  bad answer and timeout, auto-revert, shadow, feedback, auth); fixture reproducibility;
  and a scan for ids, contact details and secrets in the tree.
- `python -m unittest` in `byom/`: 5 tests for the Cog predictor's validation, weights
  overlay and I/O. Clef itself isn't loaded.
- `npm run typecheck` for the CLI and the Worker, and `wrangler deploy --dry-run` to
  bundle the Worker.
- [detect-secrets](https://github.com/Yelp/detect-secrets) 1.5.0 over the tree and every
  commit, with no findings.

**Not tested:** no call has been made to Cloudflare from this repo, no Clef weights have
been loaded, no Cog image has been built, and no LoRA has been trained. Those steps need
an account, a GPU or gated access, as listed above.

## Layout

```
src/        CLI and library (capture, redact, split, dataset, arms, evaluate, metrics, lora, byom, canary)
worker/     canary Worker: handler.ts (logic), index.ts (wiring and the Durable Object)
byom/       Cog package for a tuned Clef (cog.yaml, predict.py, tests)
fixtures/   synthetic decisions, regenerated by scripts/make-fixtures.ts
docs/       SOURCES, GAPS, MEASURING
test/       node:test suites
```

## License

MIT. Clef weights are Apache 2.0 from Cloudflare and aren't included here.
