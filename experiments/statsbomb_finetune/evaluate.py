"""Unlock test evaluation only after both supervised arms have completed."""
import csv
import json
import math
from pathlib import Path
import time

from common import OUT, read, save, sha, freeze, metrics, predict
from train import (preflight, check_training_protocol, load_split, build_model, probabilities)


def brier_cluster_interval(rows, p, baseline, seed=2026111303):
    import numpy as np
    mids = sorted({r["match_id"] for r in rows})
    if len(mids)<20:
        raise ValueError("Too few match groups for the planned exploratory interval")
    grouped = []
    for mid in mids:
        indexes = [i for i,r in enumerate(rows) if r["match_id"]==mid]
        delta = sum((p[i]-rows[i]["label"])**2-(baseline[i]-rows[i]["label"])**2 for i in indexes)
        grouped.append((delta,len(indexes)))
    rng = np.random.default_rng(seed)
    groups = np.array(grouped)
    draws = rng.integers(0,len(groups),size=(5000,len(groups)))
    sums = groups[draws].sum(axis=1)
    # Two primary contrasts: Bonferroni-adjusted 97.5% intervals.
    interval = np.quantile(sums[:,0]/sums[:,1],[.0125,.9875]).tolist()
    difference = sum(d for d,_ in grouped)/sum(n for _,n in grouped)
    return {"brier_difference":difference,"match_bootstrap_97_5":interval,"matches":len(mids),
            "interpretation":"Exploratory conditional comparison; earlier model/prompt work used these sources."}


def main():
    if (OUT/"test_evaluation.json").exists():
        print("Existing fixed test evaluation:",json.dumps(read(OUT/"test_evaluation.json")["scores"],indent=2))
        return
    protocol = check_training_protocol()
    if not {"head","lora"}.issubset({p.name for p in OUT.iterdir()}):
        raise ValueError("Both supervised training arms must complete before opening the test split")
    manifests = {arm:read(OUT/arm/"manifest.json") for arm in ("head","lora")}
    for arm, manifest in manifests.items():
        if manifest["status"] != "completed" or manifest["protocol_sha256"] != sha(OUT/"protocol.json") or manifest["smoke_only"]:
            raise ValueError("Both arms must have completed on the same frozen protocol")
        if manifest["code_sha256"]["train.py"] != sha(Path(__file__).with_name("train.py")):
            raise ValueError("Training implementation changed since checkpoint creation")
        if manifest["code_sha256"]["runtime.py"] != sha(Path(__file__).with_name("runtime.py")):
            raise ValueError("Runtime implementation changed since checkpoint creation")
    if manifests["head"]["base_revision"] != manifests["lora"]["base_revision"]:
        raise ValueError("Pretrained revisions differ between arms")
    if preflight()["status"] != "runtime_ready_weights_not_yet_checked":
        raise ValueError("Training runtime unavailable for evaluation")
    import torch
    from peft import PeftModel
    # Test labels are first parsed by this separate evaluation command.
    rows = load_split("test",protocol)
    prior = read(OUT/"baseline_models.json")["training_goal_rate"]
    predictions = {}
    for arm in ("head","lora"):
        selection = read(OUT/arm/"best/selection.json")
        checkpoint = OUT/arm/"best"
        if selection["head_sha256"] != sha(checkpoint/"head.pt"):
            raise ValueError("Selected probability head changed")
        for relative, expected in selection["adapter_sha256"].items():
            if sha(checkpoint/relative) != expected:
                raise ValueError("Selected adapter changed")
        backbone,head,processor,revision = build_model("head",prior,revision=manifests[arm]["base_revision"])
        if arm=="lora":
            backbone = PeftModel.from_pretrained(backbone,checkpoint/"adapter",is_trainable=False)
        head.load_state_dict(torch.load(checkpoint/"head.pt",map_location="cuda",weights_only=True))
        predictions[arm] = probabilities(backbone,head,processor,rows)
        del backbone,head,processor
        torch.cuda.empty_cache()
    by_id = {r["shot_id"]:r for r in read(OUT/"selected.json")["test"]}
    reference_rows = [by_id[r["shot_id"]] for r in rows]
    baseline = read(OUT/"baseline_models.json")
    predictions["training_goal_rate"] = [prior]*len(rows)
    for name,model in baseline["models"].items():
        predictions[name] = predict(model,reference_rows)
    predictions["statsbomb_xg"] = [r["statsbomb_xg"] for r in reference_rows]
    scores = {name:metrics([r["label"] for r in rows],p) for name,p in predictions.items()}
    contrasts = {name:brier_cluster_interval(rows,predictions["lora"],predictions[name])
                 for name in ("head","image_feature_logistic")}
    result = {"scores":scores,"primary_contrasts_lora_minus_reference":contrasts,
        "protocol_sha256":sha(OUT/"protocol.json"),"evaluated_unix":time.time(),
        "paid_api_calls":0,"cohort":"Same 500 held-out shots; no output calibration or averaging."}
    freeze(OUT/"test_evaluation.json",result)
    with (OUT/"test_predictions.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=["shot_id","match_id","goal"]+list(predictions))
        writer.writeheader()
        for i,row in enumerate(rows):
            writer.writerow({"shot_id":row["shot_id"],"match_id":row["match_id"],"goal":row["label"],
                             **{name:p[i] for name,p in predictions.items()}})
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
