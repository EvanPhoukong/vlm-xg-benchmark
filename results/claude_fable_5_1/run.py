"""Reproduce the original 300-shot StatsBomb run with Claude Fable 5.1."""
import argparse
import base64
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import time

import anthropic
import pandas as pd
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from xg_vlm_pilot import PROMPT, bootstrap_brier_diff, calibration_plot, parse, scores

MODEL = "claude-fable-5-1"
MAX_TOKENS = 600  # Same as the original Anthropic request.
CAP = 10
SOURCE = ROOT / "shots_opus5_5.csv"
PROMPT_TEXT = PROMPT.format(context="")


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def inputs():
    data = pd.read_csv(SOURCE)
    if len(data) != 300 or data.shot_id.nunique() != 300 or data.goal.sum() != 42:
        raise RuntimeError("Original cohort failed the 300-shot / 42-goal check.")
    requests = {}
    images = {}
    for shot_id in data.shot_id:
        image = ROOT / "results" / "images" / f"{shot_id}.png"
        with Image.open(image) as img:
            if img.size != (539, 421):
                raise RuntimeError(f"Unexpected image dimensions for {shot_id}")
        content = image.read_bytes()
        images[shot_id] = sha256(content).hexdigest()
        requests[shot_id] = {
            "custom_id": shot_id,
            "params": {
                "model": MODEL, "max_tokens": MAX_TOKENS, "inference_geo": "global",
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": base64.b64encode(content).decode(),
                    }},
                    {"type": "text", "text": PROMPT_TEXT},
                ]}],
            },
        }
    manifest = {
        "model": MODEL, "max_tokens": MAX_TOKENS, "inference_geo": "global",
        "transport": "Anthropic Message Batches API", "budget_usd": CAP,
        "competition": 43, "season": 106, "seed": 7,
        "cohort_source": SOURCE.name, "cohort_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
        "prompt": PROMPT_TEXT, "image_sha256_by_shot": images,
    }
    manifest_path = OUT / "inputs.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("The saved cohort, prompt, or image inputs changed.")
    save(manifest_path, manifest)
    return data, requests


def run_batch(client, name, requests, input_bound, prior_cost):
    result_path = OUT / f"{name}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    ids = {request["custom_id"] for request in requests}
    bound = len(ids) * (5 * input_bound + 25 * MAX_TOKENS) / 1_000_000
    state_path = OUT / f"{name}_submission.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if set(state["shot_ids"]) != ids or not state["batch_id"]:
            raise RuntimeError(f"Reconcile {state_path.name} before resubmitting.")
        batch_id = state["batch_id"]
    else:
        if prior_cost + bound > CAP:
            raise RuntimeError(f"Batch would exceed the ${CAP} cap.")
        state = {"shot_ids": sorted(ids), "reserved_usd": bound, "batch_id": None}
        save(state_path, state)
        batch_id = client.messages.batches.create(requests=requests).id
        state["batch_id"] = batch_id
        save(state_path, state)
    print(f"{name} {batch_id}: {len(ids)} requests, reserved ${bound:.4f}", flush=True)
    last_counts = None
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.request_counts != last_counts:
            print(batch.processing_status, batch.request_counts, flush=True)
            last_counts = batch.request_counts
        if batch.processing_status == "ended":
            break
        time.sleep(20)
    results = {}
    for entry in client.messages.batches.results(batch_id):
        result = entry.result
        row = {"status": result.type, "cost_usd": 0}
        if result.type == "errored":
            row["error_type"] = result.error.error.type
            row["error_message"] = result.error.error.message
        if result.type == "succeeded":
            msg = result.message
            usage = msg.usage
            row.update(model=msg.model, stop_reason=msg.stop_reason,
                       input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                       cost_usd=(5 * usage.input_tokens + 25 * usage.output_tokens) / 1_000_000)
            if msg.model == MODEL:
                output = "".join(block.text for block in msg.content if block.type == "text")
                xg, reason = parse(output)
                if pd.notna(xg):
                    row.update(vlm_xg=xg, vlm_reason=reason)
        results[entry.custom_id] = row
    if set(results) != ids:
        raise RuntimeError("Batch did not return all requested IDs.")
    save(result_path, results)
    return results


def report(data, predictions, spend):
    data["vlm_xg"] = data.shot_id.map({k: v["vlm_xg"] for k, v in predictions.items()})
    data["vlm_reason"] = data.shot_id.map({k: v["vlm_reason"] for k, v in predictions.items()})
    data["gap"] = data.vlm_xg - data.sb_xg
    data.to_csv(OUT / "shots.csv", index=False)
    columns = ["vlm_xg", "sb_xg", "baseline_xg"]
    calibration_plot(data, columns, OUT / "calibration.png")
    lines = [
        "StatsBomb 2022 FIFA World Cup (competition 43, season 106)",
        f"Original 300-shot Claude cohort | {data.goal.sum()} goals | image-only, zero-shot",
        f"Claude {MODEL} | Anthropic Message Batches API | max_tokens={MAX_TOKENS}",
        f"Usage-based API cost ${spend:.4f} (cap ${CAP:.2f})", "",
        f"{'model':<16}{'AUC':>8}{'Brier':>9}{'LogLoss':>9}{'MeanXG':>9}",
    ]
    for column in columns:
        s = scores(data.goal, data[column])
        lines.append(f"{column:<16}{s['auc']:>8.3f}{s['brier']:>9.4f}"
                     f"{s['logloss']:>9.4f}{s['mean_pred']:>9.3f}")
    for column in columns[1:]:
        gap, (lo, hi) = bootstrap_brier_diff(data.goal, data.vlm_xg, data[column])
        lines.append(f"Brier(Fable) - Brier({column}) = {gap:+.4f} "
                     f"(95% bootstrap CI {lo:+.4f} to {hi:+.4f})")
    summary = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(summary)
    print(summary, end="")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="Verify local inputs without API calls.")
    args = ap.parse_args()
    with (OUT / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data, requests = inputs()
        print(f"Verified {len(data)} original shots and images.", flush=True)
        if args.check:
            return
        key, = set(re.findall(r"\bsk-ant-[a-zA-Z0-9_-]+", (ROOT.parent / "keys.txt").read_text()))
        spend = 0
        predictions = {}
        ids = list(requests)
        with anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com",
                                max_retries=0, timeout=90) as client:
            # All images have the same dimensions and the prompt is identical.
            request = requests[ids[0]]["params"]
            tokens = client.messages.count_tokens(model=MODEL, messages=request["messages"])
            input_bound = tokens.input_tokens + 256
            for name, selected in (("first", ids[:10]), ("main", ids[10:]), ("retry", ids)):
                batch_ids = [sid for sid in selected if sid not in predictions]
                if not batch_ids:
                    continue
                results = run_batch(client, name, [requests[sid] for sid in batch_ids],
                                    input_bound, spend)
                spend += sum(row["cost_usd"] for row in results.values())
                predictions.update({sid: row for sid, row in results.items() if "vlm_xg" in row})
                print(f"{len(predictions)}/{len(ids)} scored, ${spend:.4f} used", flush=True)
                if name == "first" and len(predictions) < 8:
                    raise RuntimeError("Fewer than 8 of the first 10 requests produced usable predictions.")
        if len(predictions) != len(ids):
            raise RuntimeError(f"Only {len(predictions)} of {len(ids)} shots were scored.")
        report(data, predictions, spend)


if __name__ == "__main__":
    try:
        main()
    except anthropic.APIError as exc:
        raise SystemExit(f"Anthropic {type(exc).__name__}, status {getattr(exc, 'status_code', None)}") from None
