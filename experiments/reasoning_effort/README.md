# Direct Luna reasoning-effort pilot

Executed 29 October 2026 after user approval. Twelve previously studied World Cup shots, four per distance stratum, each received one low-effort and one medium-effort request. Original images and zero-shot direct-xG prompt were unchanged. Only gpt-6-luna was used. All 24 requests completed; there were no generation retries or automatic extensions.

Results: ../../results/reasoning_effort_v1/report.md
Protocol, selected IDs, raw responses, token-count preflights, usage, predictions and integrity checks are preserved in that directory. The estimated generation cost was $0.0063094 at published standard Luna rates. This is not a billing statement or account-balance check.

The runner uses Python standard library HTTP requests, with no SDK dependency or hidden retries. NumPy is needed only for the offline paired match bootstrap. API credentials are read privately from OPENAI_API_KEY or the existing authorized keys file. No key values are printed or stored in results.

Offline verification:

    python -m unittest discover -s experiments/reasoning_effort -p 'test_*.py' -v
    python experiments/reasoning_effort/verify_completed.py

The default runner prepares/verifies a frozen protocol without making API requests. --report rebuilds the basic report from a complete cohort; verify_completed.py adds the qualitative factual review. The report's pilot-fragility paragraph records an additional post-hoc sensitivity observation.

Do not overwrite the frozen protocol, rerun requests for a completed cohort, or treat these 12 shots as independent confirmation. Any expansion or additional reasoning settings require a new concrete proposal and budget.
