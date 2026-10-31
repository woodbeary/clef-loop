# Sources

Every Cloudflare behavior clef-loop relies on, where it's documented, and what the code
does with it. All pages were read on 2026-10-31. If a page has changed since, trust the
page and open an issue.

## Clef

| Source | What clef-loop takes from it |
| --- | --- |
| [Clef-flash model page](https://developers.cloudflare.com/workers-ai/models/clef-flash/) and its [input](https://developers.cloudflare.com/workers-ai/models/clef-flash/schema-input.json) / [output](https://developers.cloudflare.com/workers-ai/models/clef-flash/schema-output.json) schemas | The System One request (`model`, `state`, `questions`) and response (`answers`, `usage`). The limits in `src/schema.ts`: 1-64 questions, id pattern, 2-255 choice options, 2-10 score levels. The selector values `clef` and `clef-flash`. |
| [Clef model page](https://developers.cloudflare.com/workers-ai/models/clef/) | `@cf/cloudflare/clef`, same API. |
| [Changelog, 2026-10-01](https://developers.cloudflare.com/changelog/post/2026-10-01-clef-workers-ai/) | Clef runs through the Workers AI binding or `/ai/run`, and works with AI Gateway. Fine-tuning is offered as an RL service with design partners. |
| [Launch post](https://blog.cloudflare.com/clef-decision-models/) | The intended loop: capture with AI Gateway, fine-tune, redeploy with "Workers AI + BYO Model", built on BYOM (Cog). Fine-tuning starts as a hands-on service with Cloudflare's FDE team. Clef trains a joint head with rank-256 adapters on a frozen Qwen backbone. |
| [Clef-flash on Hugging Face](https://huggingface.co/Cloudflare/clef-flash) | Open weights (Apache 2.0), Qwen3.5-9B backbone, `joint_schema_model.py` with `load_release_model` and `systemone`, tested with torch 2.11 and transformers 5.10.2, and SGLang serving `/v1/systemone`. Used by `byom/predict.py` and the `systemone-url` arm. |

## Workers AI fine-tunes (LoRA)

| Source | What clef-loop takes from it |
| --- | --- |
| [Using LoRA adapters](https://developers.cloudflare.com/workers-ai/features/fine-tunes/loras/) | File names, the 300 MB limit, rank up to 32, `model_type` of mistral, gemma or llama, `wrangler ai finetune create`, the REST endpoints `POST /ai/finetunes` and `/finetune-assets`, and inference with `lora` and `raw: true`. Checked by `src/lora.ts`. |
| [Fine-tune with AutoTrain](https://developers.cloudflare.com/workers-ai/guides/tutorials/fine-tune-models-with-autotrain/) | The training data format: a CSV with one column, `text`, each row in the base model's chat template. The supported base models. `lora-r` 8 and no quantization. Written by `src/dataset.ts`. |
| [Mistral 7B Instruct v0.2 LoRA model page](https://developers.cloudflare.com/workers-ai/models/mistral-7b-instruct-v0.2-lora/) | The model id `@cf/mistral/mistral-7b-instruct-v0.2-lora` and its `prompt`, `raw`, `lora` inputs. |

## Bring Your Own Model

| Source | What clef-loop takes from it |
| --- | --- |
| [The AI platform post](https://blog.cloudflare.com/ai-platform/) (2026-04-16, updated 2026-07-15) | BYOM packages a model with Replicate's Cog (`cog.yaml` and a Python predictor), `cog build`, then a push to Workers AI. The customer-facing APIs and wrangler commands were still being built and tested with design partners. Hence `byom/` and the dry-run plan in `src/byom.ts`. |
| [Replicate joins Cloudflare](https://blog.cloudflare.com/replicate-joins-cloudflare/) | Custom models and fine-tunes on Workers AI, using Cog. |
| [Cog](https://github.com/replicate/cog) | The `cog.yaml` keys and the `BasePredictor` / `Input` predictor interface. |

## AI Gateway

| Source | What clef-loop takes from it |
| --- | --- |
| [Workers bindings](https://developers.cloudflare.com/ai-gateway/usage/worker-binding-methods/) | `env.AI.run(model, input, { gateway: { id, metadata } })`, `env.AI.aiGatewayLogId`, and `env.AI.gateway(id).patchLog(logId, { feedback, score, metadata })`. Used by `worker/handler.ts`. |
| [Custom metadata](https://developers.cloudflare.com/ai-gateway/observability/custom-metadata/) | Up to five entries per request, with string, number or boolean values. The `cf-aig-metadata` header. Keys starting with `cf.` are reserved. |
| [Add human feedback using the API](https://developers.cloudflare.com/ai-gateway/evaluations/add-human-feedback-api/) | `GET .../ai-gateway/gateways/{id}/logs` and `PATCH .../logs/{id}`, `feedback` of 1 or -1, and log ids that are opaque strings (UUIDv7 or ULID). |
| [Logs API reference](https://developers.cloudflare.com/api/resources/ai_gateway/subresources/logs/) | `per_page` up to 50, `order_by`, and the `/logs/{id}/request` and `/logs/{id}/response` endpoints. Their bodies are typed `unknown`. Used by `src/capture.ts`. |
| [Logging](https://developers.cloudflare.com/ai-gateway/observability/logging/) | Logs are on by default. `cf-aig-collect-log` and `cf-aig-collect-log-payload` turn them off per request. Gateways created on or after 2026-09-24 follow Workers Logs pricing and retention. |
| [REST API](https://developers.cloudflare.com/ai-gateway/usage/rest-api/) | `POST /ai/run/{model}` with `cf-aig-gateway-id` for Workers AI models, and `POST /ai/v1/chat/completions` for third-party models such as `openai/gpt-5.5`, billed through Unified Billing. Used by `src/cloudflare.ts`. |

## Workers platform

| Source | What clef-loop takes from it |
| --- | --- |
| [Durable Objects](https://developers.cloudflare.com/durable-objects/) | One instance per route version holds the revert window. An instance handles one request at a time, so the window can't race. |
| [workers.dev](https://developers.cloudflare.com/workers/configuration/routing/workers-dev/) | A workers.dev hostname isn't in your zone, so zone WAF rules don't apply to it. See the note in `wrangler.jsonc`. |
