# Direct visual xG model search

Authorized 29 October 2026 by the user's request to continue autonomous direct-xG experiments under the provider limits. This campaign uses a stricter $3 combined incremental ceiling. No automatic future campaign is created.

Five fixed conditions: original Luna low; Luna low with missing-information instructions; original Terra medium; Terra medium with the same instructions; original Sol low. This is a new-model and prompt study, not another claim that low/medium/high effort is untried. Gemini/Claude credentials were unavailable; neither provider is charged.

All VLM xG remains raw. No calibrator, trained probability head, numeric query geometry, baseline value, known outcome, few-shot probability examples, ensemble, tools or fine-tuning enters inference. The generic uncertainty instruction adds no goal rate or numerical anchors.

Development reuses 50 AFCON shots. The independent 200-shot La Liga 2015/16 cohort is selected from new whole match groups before any inference. All source and selection hashes are preserved. Select one winner on the complete development screen, then freeze and compare it against new original-Luna responses, the original frozen four-feature logistic, StatsBomb and a separately identified no-header logistic. Two primary Brier comparisons use 97.5% whole-match intervals. No test-set-driven prompt changes or extensions.

Use the repository Python environment. Preparation is read-only with respect to provider generation; it freezes files and downloads public data. Paid stages are explicit:

```
python -m unittest discover -s experiments/direct_model_search -p 'test_*.py' -v
python experiments/direct_model_search/run.py prepare
python experiments/direct_model_search/run.py prepare-fresh
python experiments/direct_model_search/run.py run-screen
python experiments/direct_model_search/run.py score-screen
python experiments/direct_model_search/run.py confirm
python experiments/direct_model_search/run.py run-confirmation
python experiments/direct_model_search/run.py score-confirmation
python experiments/direct_model_search/report.py
```

The engine refuses changed frozen code, selection, images or candidate. It records every attempt before dispatch and includes concurrent full-output reservations in the cap. Unknown/failed/invalid responses stop further dispatch; no hidden retries or replacement shots. Resume only a fully reconciled prefix. Offline audit reconstructs every request and compares its raw response and usage, checks prior-match exclusion and scans privately for exact credential leakage. No credentials are printed or stored in artifacts.

Output: `results/direct_model_search_v1/REPORT.md`, prediction tables, raw responses, protocols, source/image hashes and `audit.json`. Cost is usage-derived, not a verified live balance or bill for all account activity.
