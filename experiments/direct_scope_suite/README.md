# Direct image-only Luna experiment suite

Completed four fixed studies with unchanged zero-shot direct xG prediction. Start with ../../results/direct_scope_suite/RESEARCH_SUMMARY.pdf or FINAL_REPORT.md. Current results and scope are summarized in ../../CURRENT_RESEARCH_STATUS.md.

The API engine uses only the standard library, with a shared persistent ledger, input-token preflights, full next-response reservations, a $2 campaign working cap, no automatic retries and no model fallback. Credentials are read privately from the existing authorized environment or keys file; never print or copy them into artifacts.

The baseline/statistical/rendering environment is the repository's existing WSL virtual environment. The inference and PDF runtime is the bundled Windows Python. No software was installed for this study.

Offline reproduction, from the repository:

    .venv/bin/python -m unittest discover -s experiments/direct_scope_suite -p 'test_*.py' -v
    .venv/bin/python experiments/direct_scope_suite/analyze.py reasoning
    .venv/bin/python experiments/direct_scope_suite/analyze.py rendering
    .venv/bin/python experiments/direct_scope_suite/analyze.py confirmation
    .venv/bin/python experiments/direct_scope_suite/analyze.py invariance
    .venv/bin/python experiments/direct_scope_suite/verify_final.py
    .venv/bin/python experiments/direct_scope_suite/build_report.py

make_summary_pdf.py uses reportlab in the bundled Windows runtime. Both PDF pages were rendered and visually checked after final layout repair. The marker preview and effect-interval figure were also inspected.

Do not clear the ledger or overwrite frozen files to repeat paid calls. A completed run_stage.py schedule makes no additional model requests. Public cached data, source hashes, match splits, selected IDs, raw responses and per-shot predictions are preserved for review. The four-feature baseline includes header information absent from the image; a no-header control is reported separately on fresh cohorts.
