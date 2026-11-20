# Changes aimed at core direct-xG errors

Authorized after the user requested fixes for probability overestimation, unreliable visual behavior and missing input context. The user confirmed no authorized local footage is available. This campaign uses open data and an incremental $4 ceiling, within the standing $10 provider limits. Costs are estimates, not live account balances.

Four fixed methods: original Luna low; Astra low with identical zero-shot input; Luna with 48 outcome-labelled historical images; and a control using the same images with permuted labels and the same positive count. The fixed natural random 48-image World Cup training sample has four goals. Earlier work supplied four geometry examples only, never outcomes. The new condition is few-shot/in-context learning, explicitly separate from zero-shot and from weight fine-tuning. No calibrated xG example targets or numerical query answers enter the requests.

A separate motion study uses only actual Metrica tracking at event frame offsets -25, -12 and 0 (25 Hz), never future frames or end positions. Three of the 46 prior pilot shots lack required past actor/ball tracking, leaving all 43 eligible shots. Matched single-frame/history conditions share the exact final image and generic prompt; the original snapshot is a separate reference. Twelve unchanged repeats test variability without averaging predictions. This is reused two-game diagnostic data, not new independent accuracy evidence.

The main 200-shot pool was reserved from 34 unused La Liga 2016/17 match groups, excluding all 488 prior groups, before any inference. A fixed 12-shot by four-arm reused-AFCON pilot measures usage/validity only, with no accuracy selection. The prespecified conservative usage formula selected the first 150 shots; no model/prompt was chosen from pilot accuracy. Main comparisons are Astra minus Luna, Astra minus original four-feature logistic, examples minus Luna, examples minus original logistic, and examples minus shuffled labels. Five 99% whole-match intervals control the five primary comparisons within this study. Separate AUC, short-range errors, constant references, exact repeats and controlled visual probes prevent a mean-only improvement from being promoted as a core fix.

All output probabilities remain direct raw answers. No recalibration, trained statistical VLM head, blends, output averaging, enforced consistency or fine tuning. An improved in-context method must be identified as such. Historical query labels and baseline predictions never enter any request; historical training outcomes intentionally enter only the two example-based arms. Actual model outputs, settings, source/image/request hashes, counts and usage are retained. Unknown charges, invalid output or a cap stop dispatch; no hidden retries or replacement shots. Full reservations for concurrent requests are included, and the runner waits for in-flight charges to reconcile rather than raising its cap.

Offline preparation and controls (do not overwrite existing frozen artifacts):

```
python experiments/core_fixes/prepare.py
python -m unittest discover -s experiments/core_fixes -p 'test_*.py' -v
```

Explicit fixed paid stages:

```
python experiments/core_fixes/run.py pilot
python experiments/core_fixes/run.py plan-main
python experiments/core_fixes/run.py main
python experiments/core_fixes/run.py repeats
python experiments/core_fixes/run.py probes
python experiments/core_fixes/run.py motion
```

The exact pilot attempts file is preserved in `pilot/attempts_snapshot.json` before later attempts are appended, and its hash must match `resource_choice.json`.

Offline scoring/reporting:

```
python experiments/core_fixes/analyze.py
python experiments/core_fixes/report.py
```

Outputs: `results/core_fixes_v1/REPORT.md`, full `summary.json`, `audit.json`, per-stage raw responses and predictions, frozen protocols, resource choice, examples, source data and original query images. Reader illustrations are separate files and never change model inputs. The 34 reserved match groups are considered used in future research (522 recorded StatsBomb groups total).
