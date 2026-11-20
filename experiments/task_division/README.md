# Luna task division and image/event fusion

Uses the existing 200-shot Bundesliga cohort, unchanged diagrams, saved pre-shot
event metadata and frozen probability heads. No new dataset or video is added.
The user authorized OpenAI Luna only: requests must use gpt-6-luna, with no
fallback to another OpenAI model or provider.

Each shot receives three independent, sequential requests in randomized order:

1. A new monolithic geometry-only control.
2. A shooter/goalkeeper specialist.
3. A defender specialist.

Code concatenates specialist measurements and calculates geometry. Both visual
conditions are scored by frozen image-only and geometry/context probability
heads. A training-only metadata baseline and pixel/known-position controls help
determine whether images add information. There are 600 successful responses in
total. Specialist inference uses two calls per shot versus one for the control;
this is not an equal-cost comparison or a comparison of different base models.

The two primary paired Brier comparisons are specialists versus the new control,
with and without event metadata. Whole-match bootstrap 97.5% intervals adjust for
these two comparisons. Additional comparisons use secondary, unadjusted 95%
intervals. These matches were already studied: all new results are exploratory.

The VLM receives only an unchanged image and fixed geometry instructions. It
receives no goal, commercial xG, body part, event metadata or source coordinates.
The multimodal condition combines image measurements and structured event
metadata downstream in the supervised head. It cannot recover video-only cues.

Old experiments, caches and models are read only. New outputs default to:

/mnt/c/Users/ephou/Downloads/xG/output/task-division-v1

Run in the existing repository WSL virtual environment:

    python -m unittest discover -s experiments/task_division -p 'test_*.py' -v
    python experiments/task_division/run_experiment.py --prepare-only
    python experiments/task_division/replay_validation.py
    python experiments/task_division/run_experiment.py
    python experiments/task_division/run_experiment.py --offline
    python experiments/task_division/build_report.py

An exclusive process lock prevents duplicate runners. Requests use the existing
authorized environment/key handoff. Credentials never enter saved artifacts.
Protocol hashes cover code, prompts, schemas, inputs, images, probability models,
metadata and request order. Changing the protocol requires a new output folder.
Each successful response is flushed separately, and unchanged runs resume from
cache. Incomplete cohorts are not scored.

Eight targeted integrity tests passed, along with a full 200-shot evaluation and
report replay using old saved geometry and no API calls. The replay is temporary
and is not an experimental result. Completed real outputs include report.md,
comparison.png, scores.csv, predictions.csv, comparisons.csv, geometry audits,
usage.csv, integrity.json, protocol.json, and requests.jsonl.

