# Measuring it honestly

clef-loop reports nothing it didn't measure, and the numbers in this repo come from
synthetic data. This page explains how to produce numbers you can put in front of a
customer.

## Accuracy, offline

1. **Freeze the held-out set before you train.** Build once, then keep `manifest.json`.
   Its `files` hashes identify the exact split. Don't tune prompts, thresholds or the
   label schema against the held-out set. If you do, it's no longer held out.
2. **Hold out the newest decisions.** `build` does this by default (`--heldout 0.2`, or a
   count). A random split leaks the future into training. A time split asks what
   production will ask.
3. **Label with the final decision, made in the normal workflow: a person's call or a
   validator's verdict, never the model's.** Don't relabel by hand for the eval. If several
   people label, record agreement between them, and treat it as the ceiling. Record
   whether each person saw the model's suggestion first: the report's anchoring check
   splits agreement on that, and a big gap means labels lean on the model's own answers.
4. **Score every arm on the same records with the same metric.** `clef-loop eval` does
   this. A failed call or an unparseable answer counts as wrong. Report:
   - accuracy with its 95% interval, which the report includes. At 200 records and about
     80% accuracy, the interval is roughly ±5.5 points, so smaller gaps aren't evidence;
   - per-class precision and recall, because the expensive mistake is usually one class;
   - the paired McNemar p-value between arms. Two arms scored on the same records should
     be compared on their disagreements, not their totals. The report tests every pair, so
     name the comparison you care about (tuned vs base, tuned vs frontier) before the run:
     across ten pairs, the odds that some p lands below 0.05 by chance alone are far above 5%.
5. **Always include the cheap baselines.** `replay` (what the production model said at the
   time) and `majority` (always the most common label). A tuned model that barely beats
   `majority` hasn't learned the task. The trainer's dry run prints the majority baseline's
   accuracy and Brier score on the held-out set before you rent a GPU.
6. **Judge the frontier model on judgment, not format.** The chat arm asks for structured
   output limited to the allowed options. The report counts call failures and format
   failures apart; if a frontier arm's format failures are high, its accuracy is a
   parsing result, not a decision result.

## Calibration and abstention

Clef returns a probability for every allowed option. A model that says 0.9 and is right
70% of the time breaks any threshold you route on, so the report scores the
probabilities, not just the top answer:

- **Brier score** per question: the sum over options of (p - label)^2, averaged. Lower
  is better; compare arms on the same records.
- **Reliability table**: records binned by pTop into ten bins, with each bin's mean pTop
  and accuracy, and ECE (the count-weighted gap). Bins with a handful of records say
  little; read them with their counts.
- **Coverage against accuracy**: at each threshold, the share of records the model still
  decides (pTop at or above it) and its accuracy on those. Failed calls are never
  covered. Choose the threshold on training or validation data, then report it on the
  held-out set with `--abstain-below`. Picking it on the held-out set makes the held-out
  set part of training.

The Worker's `ABSTAIN_BELOW` applies the same rule online: `/decide` returns the decision
and lists the questions whose pTop fell below it in `needs_person`.

## Accuracy, online

Shadow first: `MODE=shadow` serves base and runs tuned on the side, and both are logged
with their route. Then canary: `MODE=canary` with a small `CANARY_PERCENT`.

- Compare **routes over the same days**, using the human labels attached through
  `/feedback`. A before/after comparison mixes the model change with everything else
  that changed.
- Assignment is by hashed key, so it's random with respect to the decision. Keep the
  key stable per customer or ticket.
- Count fallbacks as tuned-route failures. `GET /route` shows both windows, and logs
  with `route=fallback` show each one.
- Count wrong answers too. With the validator bound (or posting to `/outcome`), a
  well-formed wrong answer is a failure in its route's window. The route reverts when the
  lower end of the 95% Wilson interval for tuned's failure rate is above 1%, or above
  base's own rate when that is higher. At 50 tuned requests that takes 3 failures against a
  base that never fails; one transient 429 doesn't.

## Latency

The eval's latency columns are client wall time from the machine that ran the eval,
network included. They're useful for spotting an outlier arm, and they are not
production latency. Don't quote them.

The Worker's metadata-only event (`clef_loop.decision` in Workers Logs, for every tenant)
carries `latency_ms` for the served call, measured in the Worker, with the route. That is
the number to report per route, with its window and count.

To measure production latency:
- **At the gateway:** each AI Gateway log has a `duration`. Filter by the `route`
  metadata and take p50, p95 and p99 per route over a fixed window.
- **End to end:** time from request received to decision written, in the service that
  calls the Worker. That's the number a customer feels. Report the window and the count.
- **Under load:** replay a known rate, for example 10x your busiest minute, against a
  staging route, and report the rate, duration, percentiles and drops.

## Cost

AI Gateway logs carry a `cost` per request. Cost per decision is the total cost over a
window divided by the decisions in that window, including fallback calls and the side
calls in shadow mode. Clef bills input tokens only (see the model page). Third-party
models such as `openai/gpt-5.5` bill through Unified Billing. Use invoices for any number
you publish.

## What not to claim

- Any number computed on `fixtures/`. It's synthetic, and the labels come from a rule.
- Latency from `clef-loop eval`.
- Accuracy without the count, the window and the interval.
- "Beats GPT-5.5" when the McNemar p-value is large, or when only one run was made, or
  when GPT-5.5's misses are mostly format failures.
- "Tuned Clef" for anything but a model trained on the open weights and evaluated
  offline (`systemone-url`), until it runs on Workers AI through BYOM. The report says
  which arms were self-hosted.
- "RL" for the trainer. It is supervised (cross-entropy plus Brier).
