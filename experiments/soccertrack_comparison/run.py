"""Resumable paid comparison. Sequential reservations enforce each model's own cap."""
import argparse
import base64
import json
import math
import os
from pathlib import Path
import time

from common import (ROOT, OUT, MODELS, MAX_OUTPUT, CAP, PROMPT, SCHEMA,
                    api, can_reserve, credential, digest, exposure, freeze, read, save, sha, token_cost)


def validate_answer(response, model):
    returned = response.get("model", "")
    if returned != model and not returned.startswith(model + "-"):
        raise ValueError("Unexpected returned model")
    if response.get("status") != "completed" or response.get("service_tier") not in (None, "default"):
        raise ValueError("Unexpected status or service tier")
    if response.get("reasoning", {}).get("effort") != "low":
        raise ValueError("Unexpected reasoning setting")
    texts = [part["text"] for item in response.get("output", []) if item.get("type") == "message"
             for part in item.get("content", []) if part.get("type") == "output_text"]
    answer = json.loads("".join(texts))
    if set(answer) != {"xg", "reason"} or type(answer["xg"]) not in (float, int):
        raise ValueError("Invalid answer schema")
    if not math.isfinite(answer["xg"]) or not 0 <= answer["xg"] <= 1 or not isinstance(answer["reason"], str) or not answer["reason"].strip():
        raise ValueError("Invalid probability or reason; no clipping/conversion")
    return answer


def request_body(model, image):
    return {"model": model, "store": False, "service_tier": "default",
        "reasoning": {"effort": "low"}, "max_output_tokens": MAX_OUTPUT,
        "input": [{"role": "user", "content": [
            {"type": "input_image", "detail": "high", "image_url": "data:image/png;base64," + base64.b64encode(image).decode()},
            {"type": "input_text", "text": PROMPT}]}],
        "text": {"format": {"type": "json_schema", "name": "shot_prediction", "strict": True, "schema": SCHEMA}}}


def check_protocol():
    protocol = read(OUT / "protocol.json")
    if protocol["models"] != MODELS or protocol["prompt"] != PROMPT or protocol["max_output_tokens"] != MAX_OUTPUT:
        raise ValueError("Frozen model or prompt configuration differs")
    if protocol["budget"]["per_model_cap_usd"] != CAP:
        raise ValueError("Frozen budget differs")
    if sha(OUT / "shots.json") != protocol["shots_sha256"] or sha(OUT / "source_audit.json") != protocol["sources_sha256"]:
        raise ValueError("Frozen data changed")
    if sha(ROOT / "results/soccertrack_sol_v1/manifest.json") != protocol["pilot_manifest_sha256"]:
        raise ValueError("Pilot provenance changed")
    for item in protocol["cohort"]:
        path = (ROOT / item["image"]).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != item["sha256"]:
            raise ValueError("Input image hash or containment failed")
    return protocol


def schedule(protocol, arms):
    jobs = []
    for arm in arms:
        for item in protocol["cohort"]:
            # All earlier Sol original calls have matching prompt + image fingerprints.
            if arm != "sol":
                jobs.append({"arm": arm, "condition": "original", **item})
        for item in protocol["cohort"]:
            if item["shot_id"] in protocol["consistency"]["shot_ids"]:
                jobs.append({"arm": arm, "condition": "exact_repeat", **item})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--execute", action="store_true", help="Make paid calls under the frozen per-model budget")
    parser.add_argument("--preflight", action="store_true", help="Check credential/model access only; no generation")
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("Models must be unique")
    protocol = check_protocol()
    jobs = schedule(protocol, args.models)
    prior = protocol["budget"]["prior_conservative_usd"]
    ledger_path = OUT / "ledger.json"
    ledger = read(ledger_path) if ledger_path.exists() else []
    print(json.dumps({"planned_jobs": len(jobs), "per_model_cap_usd": CAP,
        "exposure_usd": {MODELS[a]["model"]: exposure(ledger, MODELS[a]["model"], prior.get(MODELS[a]["model"], 0.)) for a in args.models},
        "mode": "execute" if args.execute else "preflight" if args.preflight else "offline_plan"}), flush=True)
    if not (args.execute or args.preflight):
        return
    # Single-writer lock covers preflight and every reservation/settlement; never break a stale lock automatically.
    lock = OUT / "runner.lock"
    try:
        handle = lock.open("x")
    except FileExistsError:
        raise SystemExit("Runner lock exists; verify the prior process and ledger before resuming") from None
    with handle:
        handle.write(json.dumps({"pid": os.getpid(), "started_unix": time.time()}))
        handle.flush()
        os.fsync(handle.fileno())
        try:
            ledger = read(ledger_path) if ledger_path.exists() else []
            if any(e["state"] != "completed" for e in ledger):
                raise ValueError("Unreconciled prior request; no paid dispatch permitted")
            key = credential()
            available = {r["id"] for r in api("models", key).get("data", [])}
            missing = [MODELS[a]["model"] for a in args.models if MODELS[a]["model"] not in available]
            if missing:
                raise ValueError("Requested models not available: " + ", ".join(missing))
            if not args.execute:
                save(OUT / "preflight.json", {"status": "passed", "generation_calls": 0, "available_requested_models": args.models})
                print("Preflight passed; no generation calls", flush=True)
                return
            engine_files = [Path(__file__), Path(__file__).with_name("common.py")]
            freeze(OUT / "execution_manifest.json", {"protocol_sha256": sha(OUT / "protocol.json"),
                "code": {p.name: sha(p) for p in engine_files}, "reused_sol_sha256": sha(OUT / "reused_sol.json")})
            completed = {e["job_id"] for e in ledger if e["state"] == "completed"}
            for job in jobs:
                job_id = f"{job['arm']}__{job['condition']}__{job['shot_id']}"
                if job_id in completed:
                    continue
                config = MODELS[job["arm"]]
                image = (ROOT / job["image"]).read_bytes()
                if __import__("hashlib").sha256(image).hexdigest() != job["sha256"]:
                    raise ValueError("Image changed after preflight")
                body = request_body(config["model"], image)
                counted = api("responses/input_tokens", key, {k: body[k] for k in ("model", "input", "text")})["input_tokens"]
                if type(counted) is not int or counted < 1:
                    raise ValueError("Invalid input-token count")
                input_bound = max(counted + 512, math.ceil(counted * 1.25))
                reserved = (input_bound * config["rates"][0] + MAX_OUTPUT * config["rates"][2]) / 1e6
                if not can_reserve(ledger, config["model"], reserved, prior.get(config["model"], 0.)):
                    save(OUT / "runner_status.json", {"status": "budget_stop", "model": config["model"], "no_request_sent": True})
                    print("Per-model budget reached; saved predictions remain resumable", flush=True)
                    return
                entry = {"job_id": job_id, "model": config["model"], "arm": job["arm"], "condition": job["condition"],
                    "shot_id": job["shot_id"], "state": "reserved", "input_count": counted, "input_bound": input_bound,
                    "charged_bound_usd": reserved, "reserved_usd": reserved, "request_sha256": digest(body),
                    "image_sha256": job["sha256"], "started_unix": time.time()}
                ledger.append(entry)
                save(ledger_path, ledger)  # Durable reservation BEFORE dispatch, including reasoning output.
                try:
                    response = api("responses", key, body)
                    save(OUT / f"responses/{job_id}.json", response)
                    usage = response.get("usage") or {}
                    estimated = token_cost(usage, config["rates"])
                    charged = token_cost(usage, config["rates"], True)
                    if usage["input_tokens"] > input_bound or usage["output_tokens"] > MAX_OUTPUT or charged > reserved:
                        raise ValueError("Usage exceeded reserved bound; stop for reconciliation")
                    answer = validate_answer(response, config["model"])
                    entry.update(state="completed", usage=usage, estimated_usd=estimated, charged_bound_usd=charged,
                                 xg=answer["xg"], reason=answer["reason"], response_sha256=sha(OUT / f"responses/{job_id}.json"))
                    save(ledger_path, ledger)
                except Exception:
                    entry["state"] = "uncertain"
                    save(ledger_path, ledger)  # Keep full reservation, including after invalid/partial responses.
                    raise
                print(f"{job_id}: complete; model exposure ${exposure(ledger, config['model'], prior.get(config['model'], 0.)):.4f}", flush=True)
            save(OUT / "runner_status.json", {"status": "completed", "new_generation_calls": len(ledger), "all_requested_jobs_complete": True})
        except Exception as error:
            save(OUT / "runner_status.json", {"status": "blocked_or_stopped", "reason": str(error),
                "new_generation_calls": len(ledger), "uncertain_requests": sum(e["state"] != "completed" for e in ledger)})
            raise SystemExit(str(error)) from None
        finally:
            # This process owns the lock; the ledger retains any unresolved reservation.
            handle.close()
            lock.unlink()


if __name__ == "__main__":
    main()
