"""Summarize completed preparation and runtime status without implying unrun model results."""
from pathlib import Path
import json

from common import OUT, read, save, check_data, sha


def main():
    protocol,data = check_data()
    runtime = read(OUT/"training_preflight.json")
    baselines = read(OUT/"baseline_validation.json")
    children = {p.name for p in OUT.iterdir()}
    arms = {arm:read(OUT/arm/"manifest.json") for arm in ("head","lora")
            if arm in children and (OUT/arm/"manifest.json").is_file()}
    both_completed = len(arms)==2 and all(m["status"]=="completed" for m in arms.values())
    test = read(OUT/"test_evaluation.json") if (OUT/"test_evaluation.json").exists() else None
    checks = {"distinct_shots":len({r["shot_id"] for rs in data.values() for r in rs})==sum(map(len,data.values())),
        "disjoint_matches":all(not ({r["match_id"] for r in data[a]}&{r["match_id"] for r in data[b]})
                                for a,b in (("train","validation"),("train","test"),("validation","test"))),
        "all_image_hashes_verified":True,
        "source_audit_unchanged":sha(OUT/"source_audit.json")==protocol["source_audit_sha256"],
        "jsonl_files_unchanged":all(sha(OUT/f"{s}.jsonl")==protocol["jsonl_sha256"][s] for s in data),
        "baseline_training_only":all(set(m["train_matches"])=={r["match_id"] for r in data["train"]}
            for m in read(OUT/"baseline_models.json")["models"].values()),
        "test_access_guard":not test or (both_completed and test["protocol_sha256"]==sha(OUT/"protocol.json"))}
    if not all(checks.values()):
        raise ValueError("Data/fitting audit failed")
    save(OUT/"audit.json",checks)
    state = ("completed" if test else "models_completed_test_pending" if both_completed else
             "data_and_baselines_prepared_training_blocked" if runtime["status"].startswith("blocked") else "prepared_pending_training")
    summary = {"status":state,
        "dataset":protocol["counts"],"eligible_cached_shots":sum(protocol["pool_counts"].values()),
        "cached_matches":protocol["cached_matches"],"training_runtime":runtime,
        "validation_baselines":baselines["scores"],"test_scored":bool(test),"finetuned_model_results":test["scores"] if test else None,"new_api_spend_usd":0}
    save(OUT/"summary.json",summary)
    lines = ["# StatsBomb image-to-xG fine-tuning pilot", "",
        ("Both supervised arms and the fixed test evaluation have completed." if test else
         "Both supervised arms completed; the fixed test evaluation remains pending." if both_completed else
         "Dataset preparation and statistical controls completed. No VLM fine-tuning has completed; no fine-tuned model accuracy is claimed."), "",
        f"The local cache contains {protocol['cached_matches']} match event files and {sum(protocol['pool_counts'].values()):,} eligible shot situations after filtering. "
        "A deterministic whole-match split within competition-season was frozen before shot outcomes were parsed. Natural random shot sampling produced:", "",
        "| Split | Shots | Goals | Matches |", "|---|---:|---:|---:|"]
    for s,v in protocol["counts"].items():
        lines.append(f"| {s} | {v['shots']} | {v['goals']} | {v['matches']} |")
    lines.extend(["", "Each shot is a new 512×512 half-pitch diagram with shooter, teammates, opponents and goalkeeper. "
        "The image whitelist excludes outcomes, StatsBomb xG, body part and all match identity. "
        "Labels are observed goal/no-goal, with unweighted binary cross-entropy and sigmoid probabilities. "
        "StatsBomb probabilities are a separate reference, not training targets.", "",
        "## Validation controls", "", "These references were fitted on training shots only. Test scoring is allowed only after both VLM arms are fixed.", "",
        "| Method | Validation Brier ↓ | AUC ↑ | Log loss ↓ |", "|---|---:|---:|---:|"])
    for name,m in baselines["scores"].items():
        lines.append(f"| {name} | {m['brier']:.5f} | {m['auc']:.3f} | {m['logloss']:.4f} |")
    lines.extend(["", "## Planned local model comparison", "",
        "1. Qwen2-VL-2B-Instruct with the 4-bit backbone frozen and a learned linear probability head.",
        "2. The same pretrained revision, quantization and initial head, plus rank-8 QLoRA on text-attention query/value projections. Vision encoder remains frozen.", "",
        "Both arms use the same images, natural class frequencies, seed, training/validation split, 3-epoch maximum, batch size 1 and gradient accumulation 8. "
        "Select only complete-epoch checkpoints using validation Brier. Separate test evaluation compares QLoRA with the frozen-backbone control and image-feature logistic baseline. "
        "Two primary Brier contrasts use 97.5% whole-match bootstrap intervals. There is no probability correction or averaging.", "",
        "## Current execution block", "", "```json", json.dumps(runtime,indent=2), "```", "",
        ("The host reports an NVIDIA GeForce RTX 5060 with 8 GB. The accessible Windows Python lacks the training stack. "
        "Launching the Linux environment with WSL returned Wsl/E_ACCESSDENIED. Earlier network preflight also failed. "
        "The available Ollama weights are quantized inference artifacts, not a Transformers training checkpoint. "
        "No package installation, model download, cloud job or paid API generation completed. New API spending is $0."
        if runtime["status"].startswith("blocked") else "A compatible local CUDA runtime is available. See per-arm manifests for actual training completion. No cloud rental or paid API is used."), "",
        "## Reproduce", "", "From the vlm-xg-benchmark repository root, with NumPy/Pillow:", "", "```bash",
        "python experiments/statsbomb_finetune/prepare.py",
        "python experiments/statsbomb_finetune/baselines.py",
        "python -m unittest discover -s experiments/statsbomb_finetune -p 'test_*.py' -v",
        "python experiments/statsbomb_finetune/train.py", "```", "",
        "In a separate compatible Python/CUDA training environment, install the dependencies described in experiments/statsbomb_finetune/README.md. "
        "Run a genuine one-batch forward/backward smoke test first, then the two full arms. Downloads are public pretrained weights, with no paid service calls:", "",
        "```bash",
        "python experiments/statsbomb_finetune/train.py --execute --arm lora --smoke --allow-download",
        "python experiments/statsbomb_finetune/train.py --execute --arm head",
        "python experiments/statsbomb_finetune/train.py --execute --arm lora",
        "python experiments/statsbomb_finetune/evaluate.py", "```", "",
        "## Interpretation and provenance", "",
        "These are reused source matches from earlier experiments. The new train/validation/test groups do not overlap, but the pilot is exploratory rather than untouched independent confirmation. "
        "Training on diagrams also does not establish transfer to real broadcast frames. The filtered pool excludes penalties, invalid/out-of-half shooters, missing/invalid freeze frames, unrecorded opposing keepers, unknown outcomes and missing reference xG; per-match exclusion counts are retained. "
        "Distances are StatsBomb normalized pitch-coordinate units, not verified physical metres. The image-feature logistic uses distance, opening angle and opponents in the shooting cone; the secondary header model has extra explicit body-part information.", "",
        "Data source: StatsBomb Open Data, https://github.com/hudl/open-data. Retain StatsBomb attribution and follow the repository license. "
        "The new renderer differs from the original study renderer; older zero-shot scores are not inserted into this new comparison.", "",
        "Files: protocol.json, match_plan.json, source_audit.json, selected.json, train.jsonl, validation.jsonl, test.jsonl, images/, training_preview.png, baseline_models.json, baseline_validation.json, training_preflight.json, audit.json.", ""])
    if test:
        lines.extend(["## Fixed test results", "", "| Method | Brier | AUC |", "|---|---:|---:|"])
        for name,m in test["scores"].items():
            lines.append(f"| {name} | {m['brier']:.5f} | {m['auc']:.3f} |")
        lines.extend(["", "See test_evaluation.json for the two prespecified whole-match contrasts and test_predictions.csv for paired predictions.", ""])
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"status":summary["status"],"counts":summary["dataset"],"audit":checks},indent=2))


if __name__ == "__main__":
    main()
