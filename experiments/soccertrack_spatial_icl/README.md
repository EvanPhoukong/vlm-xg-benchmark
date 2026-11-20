# SoccerTrack v2: spatial diagrams with in-context examples

Combines the preceding experiment's released player positions, roles and identified-shooter diagrams with 32 labelled in-context examples. Model weights remain unchanged. Five conditions: fresh no-example diagrams, observed-outcome examples, shuffled outcome labels, spatial-logistic teacher examples and shuffled teacher labels.

Completed **1,520 predictions on all 304 eligible shots, 43 goals and ten matches**, with no failed or uncertain requests. Estimated new spending **$6.317800**; conservative **$7.448312**, below the **$10 incremental cap**. All 33 audit checks passed. Neither observed-outcome nor teacher-probability ICL established improvement over fresh no-example diagrams; all four adjusted match intervals include zero. No paid runner remains active.

Results are in `results/soccertrack_spatial_icl_v1/`. Read the [report](../../results/soccertrack_spatial_icl_v1/REPORT.md), [scores](../../results/soccertrack_spatial_icl_v1/summary.json), [frozen protocol](../../results/soccertrack_spatial_icl_v1/protocol.json) and [final audit](../../results/soccertrack_spatial_icl_v1/audit.json). The incremental limit for this study was **$10**, following the user's correction from $7.50.

## Files

| File | Purpose |
| --- | --- |
| `experiment.py` | Verify source images, select 32 other-match examples, reproduce excluded-match teacher fits, render labelled sheets and freeze the protocol |
| `run.mjs` | Count tokens, construct exact messages, cache example prefixes, reserve complete shot blocks and enforce the $10 cap |
| `test_budget.mjs` | Seven offline cost/guard checks and two whole-budget simulations, including no cache hits |
| `preflight.py` | Independently reconstruct the first 50 actual requests and verify all teacher exclusions without scoring accuracy |
| `analyze.py` | Score the terminal paired cohort, compute four adjusted match intervals and save predictions and an input preview |
| `audit.py` | Rebuild all 320 sheets and every request; verify exclusions, pairing, fixed stopping rule, costs and scores |
| `write_report.py` | Write the readable report from the audited results |

Python dependencies: NumPy and Pillow. Uses the existing helper code in `soccertrack_spatial` and `soccertrack_comparison`. Node.js uses native filesystem, crypto and `fetch`. The source diagrams and their provenance are inherited from the [completed spatial experiment](../soccertrack_spatial/README.md); no new large annotation retrieval is necessary.

## Reproduce scoring offline

From the `vlm-xg-benchmark` directory, with the saved inputs, responses and prior spatial results present:

```bash
node experiments/soccertrack_spatial_icl/test_budget.mjs
python experiments/soccertrack_spatial_icl/analyze.py
python experiments/soccertrack_spatial_icl/audit.py
python experiments/soccertrack_spatial_icl/write_report.py
```

These commands make no API requests. Analysis deliberately requires a terminal result, complete paired blocks and no active generation lock. The audit reproduces exact pixels using the preparation environment's font: 17px Arial when available, otherwise Pillow's default font. Saved sheet bytes and their hashes are the canonical inputs; different fonts or JPEG libraries can change byte-for-byte regeneration on another environment.

## Original preparation and generation

The original preparation on a fresh versioned output directory ran:

```bash
python experiments/soccertrack_spatial_icl/experiment.py prepare
```

It refuses to overwrite a frozen protocol. Preserve the existing completed study; use a new versioned experiment/output for a later investigation. Source dependencies, example identities and labels, teacher fits, image hashes and runner code are frozen. Preparation refits each spatial teacher using only the other nine matches and verifies its predictions against the preceding study's stored fit. The 32 examples are a seeded natural random sample from those same other matches, without goal balancing. Teacher labels are four-decimal training-example probabilities, not true xG or the new query's probability.

Original generation used the existing configured project-level provider key and these explicit operations in a Node module/REPL:

```javascript
const runner = await import('./experiments/soccertrack_spatial_icl/run.mjs');
await runner.setup();       // token counting and frozen bounds, before generation only
await runner.runBlock();    // explicit paid stage; repeat while the result is nonterminal
console.log(await runner.status());
```

Importing the runner and calling `status()` are read-only and make no network requests. Do not rerun `setup()` over its saved resource plan. Successful responses are never regenerated. No automatic retry occurs; uncertain requests or an interrupted block stop further dispatch for inspection.

Each block has up to ten shots and all five conditions. Its maximum input-plus-output cost is reserved durably before the first request. Reservations assume every token is uncached or newly cached at the full five-minute write rate. Completed usage is accounted at conservative prices, including actual cache reads and writes. The selected cohort is a prefix of a fixed, seeded within-match random and across-match round-robin pool. The largest affordable complete next block is selected using usage alone, before accuracy inspection. Stop when all 304 shots are complete or when the next complete five-condition shot's maximum reservation cannot fit. This guarantees complete paired shots even if caching provides no savings. Seven guard checks and two full-budget simulations cover both cached and uncached cases.

Five-minute caching marks only the last example-sheet image, sharing an identical historical-example prefix within a match/condition. The query comes after that prefix. Every request is a separate one-message call: preceding query responses never become examples. One-hour caching is not used. Cost accounting includes the provider's ordinary input, cache write/read and output token fields; no live billing balance is claimed.

## Interpretation

The primary control is a fresh no-example prediction on the same diagram, model, prompt and inference settings. Saved original-photo, diagram and photo-plus-diagram predictions are matched descriptive references. The four prespecified Brier differences compare each correctly labelled ICL condition with no examples and its own shuffled control, with 98.75% complete-match bootstrap intervals across four tests (20,000 draws). A benefit versus no examples does not establish a benefit from correct example-label pairing unless the shuffled comparison also supports it.

All ten matches were studied previously. This is annotation-assisted exploratory cross-validation with overlapping example sets and teacher training folds, not external confirmation or automatic player detection. No calibration, fitted VLM output head, averaging or fine-tuning is applied. The inherited two geometry exclusions and the source's incomplete large-file hash verification remain disclosed in the report.
