"""Recover a single JSON answer inside prose without changing the model prediction."""
import argparse
import json
from pathlib import Path

import study as s

original_parser = s.opus_answer


def parse_opus(raw):
    try:
        return original_parser(raw)
    except json.JSONDecodeError:
        if raw["stop_reason"] != "end_turn" or raw["model"] != s.MODELS["opus"]["model"]:
            raise
        text = "".join(block["text"] for block in raw["content"] if block["type"] == "text")
        decoder = json.JSONDecoder()
        candidates = []
        for offset, character in enumerate(text):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and set(value) == {"xg", "reason"}:
                candidates.append(value)
        if len(candidates) != 1:
            raise ValueError("A unique JSON xG answer is required")
        import xg_vlm_pilot
        value, reason = xg_vlm_pilot.strict_prediction(json.dumps(candidates[0]))
        return {"xg": value, "reason": reason}


def install_parser():
    s.c.freeze(s.OUT / "parser_amendment.json", {
        "reason": "One completed Claude response included explanatory prose before its single valid JSON object.",
        "scope": "Read the single xg/reason JSON object as originally returned. Probabilities, prompts, settings and cohort stay identical.",
        "additional_generation_calls": 0,
        "implementation_sha256": s.c.sha(Path(__file__)),
    })
    s.opus_answer = parse_opus


def reconcile():
    ledger = s.c.read(s.OUT / "ledger.json")
    changes = []
    for entry in ledger:
        if entry["state"] != "uncertain":
            continue
        if entry["arm"] != "opus":
            raise ValueError("This amendment only reconciles completed Claude text responses")
        response_path = s.OUT / "responses" / (entry["job_id"] + ".json")
        raw = s.c.read(response_path)
        answer = parse_opus(raw)
        usage, estimated, conservative = s.usage_cost(raw["usage"], s.MODELS["opus"]["rates"], True)
        if usage["input_tokens"] > entry["input_bound"] or conservative > entry["reserved_usd"]:
            raise ValueError("Usage must fit the original reservation")
        changes.append({"before": dict(entry), "original_response_sha256": s.c.sha(response_path)})
        entry.update(state="completed", usage=usage, estimated_usd=estimated,
                     charged_bound_usd=conservative, **answer,
                     response_sha256=s.c.sha(response_path), recovered_from_original_response=True)
    if changes:
        s.c.freeze(s.OUT / "parser_reconciliation.json", changes)
        s.c.save(s.OUT / "ledger.json", ledger)
        print("Recovered original JSON answer and reconciled its actual token usage. No new API call.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run", "analyze"])
    stage = parser.parse_args().stage
    install_parser()
    reconcile()
    if stage == "run":
        s.execute()
    else:
        s.analyze()

