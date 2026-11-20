# SoccerTrack v2 spatial-information experiment

Completed annotation-assisted xG comparison: original photographs, player-position/role diagrams, marked photographs with enlarged shot areas, and photographs plus diagrams. All four use Claude Sonnet 5.5 on the identical 304 eligible shots from ten previously studied matches. Basic and spatial logistic models use complete-match exclusions.

Read the [completed report](../../results/soccertrack_spatial_v1/REPORT.md), [scores](../../results/soccertrack_spatial_v1/summary.json) and [audit](../../results/soccertrack_spatial_v1/audit.json). All 1,216 generation requests are complete. Estimated spending $5.525214; conservative $6.683825 under the $10 experiment cap. No new calls are needed to inspect or reproduce scoring.

## Code

| File | Purpose |
| --- | --- |
| `source.mjs` | Locate the official public mirror, validate HTTP byte ranges, extract exact GSR frame records and retrieve camera resources |
| `common.py` | Parse roles/positions, match the shooter, project field coordinates and derive spatial features |
| `experiment.py` | Enforce eligibility, render inputs, fit match-held-out controls and freeze the protocol |
| `run.mjs` | Token-count planning, fixed four-condition schedule, durable request reservations, hard cap, raw responses and resume handling |
| `analyze.py` | Offline metrics, four adjusted match intervals, descriptive references, alignment sensitivity, CSV and input preview |
| `audit.py` | Offline source reconstruction, frozen hashes, no-outcome-pixel checks, fitting exclusions and request/cost reconciliation |

Python dependencies: NumPy and Pillow. Source retrieval and inference use Node.js with native `fetch`, filesystem and crypto APIs. The code imports the established statistical helpers from `experiments/soccertrack_comparison/common.py`.

## Offline reproduction

From the `vlm-xg-benchmark` directory, with the saved source cache, images and responses present:

```bash
python experiments/soccertrack_spatial/analyze.py
python experiments/soccertrack_spatial/audit.py
```

These scripts make no API calls. The audit verifies 612 exact snapshots, 20 small camera files, all 1,216 payloads/responses, logistic fits and frozen source/artifact hashes. It recreates a few images in temporary storage to confirm that changing outcomes, headers and event locations does not alter model input pixels. Scoring retains the original raw probabilities.

## Original preparation sequence

The following documents the completed study's preparation on a fresh output directory. Do not rerun preparation over this saved study or edit its frozen implementation to start a different experiment. `experiment.py prepare` deliberately refuses an existing protocol. A new investigation needs its own versioned output directory and newly frozen protocol.

The source stage used the existing audited cohort and archived pre-shot photographs in `results/soccertrack_comparison_v1/` and `results/soccertrack_extensive_v1/`, plus canonical LFS pointers in the sibling `soccertrack-v2` checkout. From a Node module or REPL in the benchmark directory:

```javascript
const fs = await import('node:fs/promises');
const source = await import('./experiments/soccertrack_spatial/source.mjs');
const frames = JSON.parse(await fs.readFile('results/soccertrack_extensive_v1/frames.json', 'utf8'));
const matches = [...new Set(frames.map(row => row.match_id))].sort();
await source.initialize();
await source.calibration(matches);
await source.retrieveMatches(matches);
```

The downloader retrieves only ranges needed for the exact pre-shot and at-event audit snapshots. Each range has a content hash and validated Content-Range. Complete large-file SHA256 verification is explicitly not claimed. Small homography and intrinsic files are verified against canonical LFS hashes. Numeric camera arrays are loaded without unpickling object arrays.

Preparation then ran:

```bash
python experiments/soccertrack_spatial/experiment.py prepare
```

The result is a frozen protocol, deterministic cohort order, eligibility/source/crop audits, input images and match-held-out statistical models. Both primary logistic models use pre-shot geometry; one adds keeper and defender features. At-event references are separate descriptive outputs of `analyze.py`.

## Completed inference sequence

The runner uses the existing project-level provider configuration in `keys.txt`; credentials are never included in result files. Importing the module makes no requests. `setup()` performs token-count requests and freezes a budget-selected natural prefix before accuracy scoring. Generation is explicit:

```javascript
const runner = await import('./experiments/soccertrack_spatial/run.mjs');
await runner.setup();                 // original pre-generation preparation only
await runner.runBatch(48);            // explicit paid stage; repeat only while incomplete
console.log(await runner.status());   // read-only status
```

For the existing completed study, use only `status()` if checking status; do not rerun `setup()`. The runner records a durable reservation before dispatch, uses six workers, enforces conservative exposure below $10 and performs no automatic retries. Uncertain or incomplete requests stop further dispatch. Successful cached predictions are never regenerated. The fixed schedule contains four conditions per shot with cyclic condition order.

## Scope

These are released-annotation-assisted predictions, not end-to-end automatic detection. Inputs use one pre-shot snapshot, no auxiliary interpolated ball track, no future snapshot, and no numeric coordinates, outcomes or reference xG in visual requests. The event actor is used to identify the shooter, not to supply its outcome. The at-event snapshot is used only for alignment auditing and the declared score-only sensitivity check. No ICL, fine-tuning, fitted VLM output correction or probability averaging was used.

All ten matches have prior research exposure. This study tests the input representation within a fixed model, and does not establish untouched generalization or isolate every contribution of coordinates, roles, explicit shooter marking and calibration. Dataset attribution and full limitations are in the report.
