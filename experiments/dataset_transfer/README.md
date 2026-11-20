# Fixed direct-VLM transfer across datasets

User-authorized continuation: try different datasets while preserving raw image-only xG and the $10 provider limits. This study enforces a stricter $1 incremental cap with full in-flight reservations.

Frozen method: original half-pitch renderer and zero-shot prompt; GPT-6 Luna low, high image detail, 4,096 maximum output/reasoning tokens. No cohort names, coordinates, body-part flags, outcomes, baseline values, target goal rates or examples enter requests. No recalibration, output correction, trained VLM head, averaging or fine tuning.

Datasets are distinguished rather than renamed as independent providers:

- StatsBomb Premier League 2015/16: 500 natural-sample test shots from 40 new match groups, 40 disjoint training matches for additional local statistical references.
- StatsBomb Women's World Cup 2019: 500 test shots from 26 new groups, 26 disjoint training matches. The unchanged VLM receives no gender/cohort metadata, an explicit limitation.
- Independent provider Metrica Sports: 46 eligible shots from public Sample Games 1 and 2, synchronized event-start tracking only. Descriptive pilot, not population-level confirmation. Anonymization prevents excluding cross-provider identity overlap.

All 356 previous StatsBomb match groups were excluded before new train/test selection. Compare raw Luna with the original frozen World Cup four-feature logistic as the primary test in the two large cohorts, using 97.5% whole-match intervals for two comparisons. Report separate same-dataset logistic fits, no-header variants, training-only constant and StatsBomb references. The Metrica comparison is descriptive because it has only two matches. No dataset shopping, goal balancing or outcome-driven extensions.

The offline goalkeeper test caught an incorrect role from a single boundary frame in Metrica. The correction fixes keeper identity from median opening-period-1 tracking and keeps it across halves; direction uses the same keeper in each opening half. All windows precede the first shot. Goalkeeper substitutions stop preparation rather than invite a guessed role. Initial failed-preflight files and images remain preserved in `metrica/` and `images/metrica/`, and are never submitted. Only `metrica_v2/` is used. `data_amendment.json` documents this pre-inference correction; the first unrun protocol remains `protocol_preinference_v1.json`.

Public source review found xGHub clips gated behind institutional access and an NDA. No access request, NDA consent, restricted-file download or third-party message was made. SkillCorner and Wyscout were assessed as alternatives but are not silently substituted as identical full-scene shot data.

Preparation and validation:

```
python experiments/dataset_transfer/audit_sources.py
python experiments/dataset_transfer/prepare.py
python -m unittest discover -s experiments/dataset_transfer -p 'test_*.py' -v
python experiments/dataset_transfer/run.py prepare
```

Explicit paid stage and offline scoring:

```
python experiments/dataset_transfer/run.py run
python experiments/dataset_transfer/analyze.py
python experiments/dataset_transfer/report.py
```

Maximum six concurrent generation calls; all reservations count toward $1. No hidden retries. Unknown charges, malformed/truncated responses, unexpected model or cap stop dispatch and preserve in-flight records. Default preparation makes no paid generation calls. No partial-cohort scoring or replacement shots.

Output: `results/dataset_transfer_v1/`. Every request is reconstructed from frozen image bytes and prompt, every raw answer/usage/identity is verified, and source, reference, selection, group separation and credential-exposure checks are retained. Report costs as usage estimates, never an account-wide bill or verified remaining balance.
