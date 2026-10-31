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
3. **Label with the human's final decision, made in the normal workflow.** Don't relabel
   by hand for the eval. If several people label, record agreement between them, and
   treat it as the ceiling.
4. **Score every arm on the same records with the same metric.** `clef-loop eval` does
   this. A failed call or an unparseable answer counts as wrong. Report:
   - accuracy with its 95% interval, which the report includes. At 200 records and about
     80% accuracy, the interval is roughly ±5.5 points, so smaller gaps aren't evidence;
   - per-class precision and recall, because the expensive mistake is usually one class;
   - the paired McNemar p-value between arms. Two arms scored on the same records should
     be compared on their disagreements, not their totals.
5. **Always include the cheap baselines.** `replay` (what the production model said at the
   time) and `majority` (always the most common label). A tuned model that barely beats
   `majority` hasn't learned the task.

## Accuracy, online

Shadow first: `MODE=shadow` serves base and runs tuned on the side, and both are logged
with their route. Then canary: `MODE=canary` with a small `CANARY_PERCENT`.

- Compare **routes over the same days**, using the human labels attached through
  `/feedback`. A before/after comparison mixes the model change with everything else
  that changed.
- Assignment is by hashed key, so it's random with respect to the decision. Keep the
  key stable per customer or ticket.
- Count fallbacks as tuned-route failures. `GET /route` shows the revert window, and logs
  with `route=fallback` show each one.

## Latency

The eval's latency columns are client wall time from the machine that ran the eval,
network included. They're useful for spotting an outlier arm, and they are not
production latency. Don't quote them.

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
- "Beats GPT-5.5" when the McNemar p-value is large, or when only one run was made.
