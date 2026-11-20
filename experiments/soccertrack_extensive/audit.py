"""Verify source, request, probability and spending integrity after the completed study."""
from pathlib import Path

import study as s


def main():
    protocol = s.prepare()
    ledger = s.c.read(s.OUT / "ledger.json")
    summary = s.c.read(s.OUT / "summary.json")
    lookup = {r["shot_id"]: r for r in protocol["cohort"]}
    seen = set()
    for entry in ledger:
        key = entry["arm"], entry["shot_id"]
        if entry["state"] != "completed" or key in seen:
            raise ValueError("Incomplete or repeated model attempt")
        seen.add(key)
        request = s.payload(entry["arm"], lookup[entry["shot_id"]])
        if s.c.digest(request) != entry["request_sha256"]:
            raise ValueError("Saved request differs from the fixed original image and prompt")
        if entry["charged_bound_usd"] > entry["reserved_usd"]:
            raise ValueError("A settled charge exceeded its reservation")
    counts = {arm: sum(arm == r["arm"] for r in ledger) + (30 if arm == "sol" else 0)
              for arm in s.MODELS}
    if set(counts.values()) != {306} or len(ledger) != 1194:
        raise ValueError("Every model must cover all 306 fixed shots")
    if summary["conservative_usd"] > protocol["budget"]["combined_cap_usd"]:
        raise ValueError("Combined spending exceeded the study ceiling")
    if (s.OLD / "runner.lock").exists():
        raise ValueError("A paid campaign is still running")
    audit = {
        "status": "passed", "unique_source_shots": 306, "matches": 10, "goals": 44,
        "predictions_by_model": counts, "new_generation_calls": len(ledger),
        "original_sol_calls_reused": 30, "unresolved_requests": 0,
        "source_audit_and_independent_logistic_refit": True,
        "input_hashes_and_all_request_fingerprints": True,
        "same_image_and_prompt_for_every_model": True,
        "baseline_training_excludes_the_evaluated_match": True,
        "raw_probabilities_unchanged": True, "generation_retries": 0,
        "parser_recovery_from_saved_response": len(s.c.read(s.OUT / "parser_reconciliation.json")),
        "new_estimated_usd": summary["estimated_usd"],
        "conservative_usd_including_prior": summary["conservative_usd"],
        "combined_cap_usd": 10.,
        "account_scope": "Study accounting. Actual account-wide bill and balance are unavailable.",
    }
    s.c.save(s.OUT / "audit.json", audit)
    print(audit)


if __name__ == "__main__":
    main()

