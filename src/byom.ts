// Step 5, gated path: Bring Your Own Model for a tuned Clef.
//
// What's public (checked 2026-10-31):
// - blog.cloudflare.com/clef-decision-models (2026-10-01): "Cloudflare Workers AI + BYO
//   Model - redeploy the fine-tuned model on Workers AI", built on "Workers AI's Bring
//   Your Own Model (Cog) work"; fine-tuning is offered first with Cloudflare's FDE team.
// - blog.cloudflare.com/ai-platform (2026-04-16): package the model with Replicate's Cog
//   (cog.yaml + a Python predictor), `cog build`, then push to Workers AI; "customer-facing
//   APIs and wrangler commands" are being built and tested with design partners.
// What isn't public: the push command, the model id a BYOM deployment gets, and how
// Workers AI maps a request onto the predictor's inputs. So clef-loop checks the Cog
// package and prints the plan; it never pretends to push.

export interface CogFiles {
  cogYaml?: string;
  predictPy?: string;
  requirements?: string;
}

export function checkCog(files: CogFiles): { errors: string[]; warnings: string[] } {
  const errors: string[] = [];
  const warnings: string[] = [];
  if (files.cogYaml === undefined) errors.push("missing cog.yaml");
  else {
    const predict = files.cogYaml.match(/^predict:\s*["']?([^"'\s]+)["']?\s*$/m);
    if (!predict) errors.push('cog.yaml needs a top-level `predict: "predict.py:Predictor"`');
    else if (predict[1] !== "predict.py:Predictor") warnings.push(`cog.yaml predicts with ${predict[1]}; clef-loop ships predict.py:Predictor`);
    if (!/^build:\s*$/m.test(files.cogYaml)) errors.push("cog.yaml needs a build: section");
    if (!/^\s+gpu:\s*true\s*$/m.test(files.cogYaml)) warnings.push("cog.yaml does not set build.gpu: true; Clef needs a GPU");
    const req = files.cogYaml.match(/^\s+python_requirements:\s*(\S+)\s*$/m);
    if (req && files.requirements === undefined) errors.push(`cog.yaml names ${req[1]}, which is missing`);
  }
  if (files.predictPy === undefined) errors.push("missing predict.py");
  else {
    if (!/class\s+Predictor\s*\(\s*BasePredictor\s*\)/.test(files.predictPy)) errors.push("predict.py needs class Predictor(BasePredictor)");
    if (!/def\s+setup\s*\(/.test(files.predictPy)) errors.push("Predictor needs setup()");
    if (!/def\s+predict\s*\(/.test(files.predictPy)) errors.push("Predictor needs predict()");
  }
  return { errors, warnings };
}

export function byomPlan(tag: string): string[] {
  return [
    "1. Put the tuned weights in byom/weights/tuned/ (files there replace the same files in the base snapshot):",
    "   trainer/train.py writes joint_head.safetensors there; fold its merged_adapters.safetensors into the",
    "   backbone shards first and put the changed shards there under their original names.",
    "   Optional: put the base snapshot in byom/weights/base/ so the image doesn't download it at startup.",
    `2. cog build -t ${tag}        # in byom/, on a machine with Docker`,
    `3. cog predict -i request="$(cat ../fixtures/systemone-request.json)"   # smoke test on a GPU box`,
    "4. Push to Workers AI: not public yet. Cloudflare runs BYOM with design partners; use the command and",
    "   model id they give you. Ask through your Cloudflare contact for BYOM access.",
    '5. Set the Worker\'s TUNED_ARM to {"kind":"systemone","model":"<the BYOM model id>","selector":"clef-flash"}',
    "   and start with MODE=shadow, then MODE=canary at a small CANARY_PERCENT.",
    "Until then, evaluate the tuned weights from a self-hosted SGLang server (documented on the Clef-flash",
    "model card) with --arm tuned=systemone-url:http://<host>:30000/v1/systemone.",
  ];
}
