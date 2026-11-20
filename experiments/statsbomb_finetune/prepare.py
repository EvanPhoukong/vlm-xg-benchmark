"""Build a supervised pilot entirely from locally cached official StatsBomb events."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random

from common import (ROOT, OUT, SEED, LIMITS, MODEL_ID, PROMPT, extract, freeze,
                    image_payload, read, render, save, sha, digest)


def inventory():
    events, metadata = {}, []
    for parent, dirs, files in os.walk(ROOT / "results"):
        dirs[:] = [d for d in dirs if d not in {"images", "frames", "responses", "__pycache__", "statsbomb_finetune_v1"}]
        for name in sorted(files):
            path = Path(parent)/name
            if name.startswith("events_") and name.endswith(".json"):
                mid = name.removeprefix("events_").removesuffix(".json")
                if not mid.isdecimal():
                    continue
                if mid in events:
                    if sha(path) != sha(events[mid]):
                        raise ValueError("Conflicting cached versions of match " + mid)
                else:
                    events[mid] = path
            elif name.endswith(".json") and (name.startswith("matches_") or name.endswith("_matches.json")):
                metadata.append(path)
    return events, sorted(metadata)


def match_metadata(paths):
    result = {}
    for path in paths:
        values = read(path)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict) or "match_id" not in item or "competition" not in item or "season" not in item:
                continue
            mid = str(item["match_id"])
            value = {"competition_id": item["competition"]["competition_id"], "season_id": item["season"]["season_id"],
                     "competition_name": item["competition"]["competition_name"], "season_name": item["season"]["season_name"]}
            if mid in result and result[mid] != value:
                raise ValueError("Conflicting competition metadata for " + mid)
            result[mid] = value
    return result


def split_matches(metadata):
    groups = defaultdict(list)
    for mid, item in metadata.items():
        groups[f"{item['competition_id']}:{item['season_id']}"].append(mid)
    plan = {"train": [], "validation": [], "test": []}
    for key, mids in sorted(groups.items()):
        mids.sort()
        random.Random(SEED + int(digest(key)[:8],16)).shuffle(mids)
        if len(mids) < 7:
            raise ValueError("Too few cached matches for a three-way competition split: " + key)
        nt, nv = int(.70*len(mids)), max(1,int(.15*len(mids)))
        for label, selected in (("train", mids[:nt]), ("validation", mids[nt:nt+nv]), ("test", mids[nt+nv:])):
            plan[label].extend(selected)
    return {s:sorted(mids) for s,mids in plan.items()}


def make_preview(data):
    from PIL import Image, ImageDraw
    examples = data["train"][:6]
    preview = Image.new("RGB", (3*300, 2*330), "white")
    draw = ImageDraw.Draw(preview)
    for i, row in enumerate(examples):
        x, y = (i%3)*300, (i//3)*330
        with Image.open(OUT/row["image"]) as im:
            preview.paste(im.resize((300,300)), (x,y))
        draw.text((x+8,y+302), f"Training example {i+1} | goal={row['goal']}", fill="black")
    preview.save(OUT/"training_preview.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"protocol.json").exists():
        from common import check_data
        protocol, data = check_data()
        print("Existing frozen data verified:", {s:len(rs) for s,rs in data.items()}, flush=True)
        return
    events, manifests = inventory()
    lookup = match_metadata(manifests)
    absent = sorted(set(events)-set(lookup))
    if absent:
        raise ValueError("Missing competition metadata for cached matches: " + str(absent))
    lookup = {mid:lookup[mid] for mid in events}
    # Freeze all match assignments before reading shot outcomes or sampling individual shots.
    plan = split_matches(lookup)
    freeze(OUT/"match_plan.json", {"seed": SEED, "splits": plan, "metadata": lookup,
        "rule": "70/15/15 whole-match split within competition-season; deterministic, no outcome stratification",
        "source": "Previously cached official StatsBomb event files; retrospective reuse, not untouched confirmation."})
    split_of = {mid:s for s,mids in plan.items() for mid in mids}
    pools = {s:[] for s in LIMITS}
    audit, seen = [], set()
    for count, (mid, path) in enumerate(sorted(events.items()), 1):
        raw = path.read_bytes()
        values = json.loads(raw)
        omitted, eligible = Counter(), 0
        item = lookup[mid]
        comp = f"{item['competition_id']}:{item['season_id']}"
        for event in values:
            row, reason = extract(event, mid, comp)
            if reason:
                omitted[reason] += 1
            if row is not None:
                if row["shot_id"] in seen:
                    raise ValueError("Repeated StatsBomb event ID across matches")
                seen.add(row["shot_id"])
                pools[split_of[mid]].append(row)
                eligible += 1
        audit.append({"match_id": mid, "relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": __import__("hashlib").sha256(raw).hexdigest(), "eligible_shots": eligible, "omissions": dict(omitted)})
        if count % 30 == 0 or count == len(events):
            print(f"Parsed cached matches {count}/{len(events)}", flush=True)
    save(OUT/"source_audit.json", audit)
    selected = {}
    (OUT/"images").mkdir(exist_ok=True)
    for i, (split, cap) in enumerate(LIMITS.items()):
        pool = sorted(pools[split], key=lambda r:r["shot_id"])
        if len(pool) < cap:
            raise ValueError(f"Insufficient {split} pool: {len(pool)} < {cap}; do not silently resize")
        selected[split] = random.Random(SEED + 100 + i).sample(pool, cap)
        for n, row in enumerate(selected[split], 1):
            name = "images/" + row["shot_id"] + ".png"
            path = OUT/name
            if not path.exists():
                render(row, path)
            row["image"], row["image_sha256"] = name, sha(path)
            row["image_input_sha256"] = digest(image_payload(row))
            if n % 500 == 0 or n == len(selected[split]):
                print(f"Rendered {split}: {n}/{len(selected[split])}", flush=True)
        with (OUT/f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for r in selected[split]:
                # No xG teacher targets, rationales, metadata or baseline predictions in model examples.
                stream.write(json.dumps({"shot_id":r["shot_id"], "match_id":r["match_id"], "image":r["image"],
                                         "image_sha256":r["image_sha256"], "label":r["goal"]}) + "\n")
    freeze(OUT/"selected.json", selected)
    protocol = {"seed": SEED, "dataset": "StatsBomb Open Data, cached event + shot freeze-frame data",
        "counts": {s:{"shots":len(rs), "goals":sum(r["goal"] for r in rs), "matches":len({r["match_id"] for r in rs})} for s,rs in selected.items()},
        "pool_counts": {s:len(rs) for s,rs in pools.items()}, "cached_matches":len(events),
        "match_plan_sha256":sha(OUT/"match_plan.json"), "source_audit_sha256":sha(OUT/"source_audit.json"),
        "selected_sha256":sha(OUT/"selected.json"), "jsonl_sha256":{s:sha(OUT/f"{s}.jsonl") for s in LIMITS},
        "code_sha256":{name:sha(Path(__file__).with_name(name)) for name in ("prepare.py","common.py")},
        "model_id":MODEL_ID, "prompt":PROMPT, "representation":"New deterministic 512x512 Pillow half-pitch renderer; identical for both supervised arms. Not the legacy benchmark renderer.",
        "model_inputs":"Only positional diagram and fixed generic prompt. No body part, numerical coordinates, outcome, xG reference, match/competition identity or previous predictions.",
        "objective":"Unweighted binary cross-entropy on observed goal/no-goal; sigmoid probability at inference. No generated numeric targets or teacher xG labels.",
        "arms":["frozen_quantized_vlm_plus_linear_head","qlora_text_attention_plus_same_linear_head"],
        "references":["training_goal_rate","image_feature_logistic","logistic_with_header","statsbomb_xg"],
        "primary_metric":"Brier score", "secondary_metrics":["log_loss","AUC","reliability","per_competition"],
        "training":{"epochs":3,"micro_batch":1,"gradient_accumulation":8,"adapter_lr":0.0002,"head_lr":0.001,
                    "weight_decay":0.01,"lora_rank":8,"lora_alpha":16,"lora_dropout":0.05,
                    "adapter_modules":"Text attention q_proj/v_proj only; vision encoder frozen",
                    "image_pixels":256*28*28,"checkpoint_selection":"lowest complete-epoch validation Brier",
                    "class_weights":False,"resampling":False,"max_seconds_per_arm":3600},
        "test_policy":"Test labels are not loaded by training/checkpoint selection; separate explicit evaluation after both arms are fixed.",
        "interpretation":"Exploratory supervised study on previously used matches. Training/validation/test are disjoint here, but this is not independent confirmation of earlier zero-shot studies.",
        "budget":{"new_paid_api_calls":0,"per_model_api_cap_usd":10,"cloud_gpu_jobs":0,"local_compute_only":True,
                  "note":"No automatic paid service or GPU rental; local compute has no token bill."},
        "provenance":{"repository":"https://github.com/hudl/open-data","credit":"StatsBomb","license":"See StatsBomb repository LICENSE.pdf; retain attribution in publications."}}
    freeze(OUT/"protocol.json", protocol)
    make_preview(selected)
    save(OUT/"data_status.json", {"status":"prepared", "counts":protocol["counts"],"pool_counts":protocol["pool_counts"], "paid_calls":0})
    print(json.dumps({"counts":protocol["counts"],"pool_counts":protocol["pool_counts"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
