# StatsBomb image-to-xG fine-tuning pilot

Dataset preparation and statistical controls completed. No VLM fine-tuning has completed; no fine-tuned model accuracy is claimed.

The local cache contains 298 match event files and 7,196 eligible shot situations after filtering. A deterministic whole-match split within competition-season was frozen before shot outcomes were parsed. Natural random shot sampling produced:

| Split | Shots | Goals | Matches |
|---|---:|---:|---:|
| train | 2000 | 181 | 207 |
| validation | 400 | 49 | 43 |
| test | 500 | 63 | 48 |

Each shot is a new 512×512 half-pitch diagram with shooter, teammates, opponents and goalkeeper. The image whitelist excludes outcomes, StatsBomb xG, body part and all match identity. Labels are observed goal/no-goal, with unweighted binary cross-entropy and sigmoid probabilities. StatsBomb probabilities are a separate reference, not training targets.

## Validation controls

These references were fitted on training shots only. Test scoring is allowed only after both VLM arms are fixed.

| Method | Validation Brier ↓ | AUC ↑ | Log loss ↓ |
|---|---:|---:|---:|
| image_feature_logistic | 0.10060 | 0.658 | 0.3591 |
| logistic_with_header | 0.09742 | 0.680 | 0.3472 |
| training_goal_rate | 0.10852 | 0.500 | 0.3775 |
| statsbomb_xg | 0.09470 | 0.726 | 0.3305 |

## Planned local model comparison

1. Qwen2-VL-2B-Instruct with the 4-bit backbone frozen and a learned linear probability head.
2. The same pretrained revision, quantization and initial head, plus rank-8 QLoRA on text-attention query/value projections. Vision encoder remains frozen.

Both arms use the same images, natural class frequencies, seed, training/validation split, 3-epoch maximum, batch size 1 and gradient accumulation 8. Select only complete-epoch checkpoints using validation Brier. Separate test evaluation compares QLoRA with the frozen-backbone control and image-feature logistic baseline. Two primary Brier contrasts use 97.5% whole-match bootstrap intervals. There is no probability correction or averaging.

## Current execution block

```json
{
  "model_id": "Qwen/Qwen2-VL-2B-Instruct",
  "missing_packages": [
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes"
  ],
  "paid_api_calls": 0,
  "cloud_gpu_jobs": 0,
  "status": "blocked_missing_training_runtime"
}
```

The host reports an NVIDIA GeForce RTX 5060 with 8 GB. The accessible Windows Python lacks the training stack. Launching the Linux environment with WSL returned Wsl/E_ACCESSDENIED. Earlier network preflight also failed. The available Ollama weights are quantized inference artifacts, not a Transformers training checkpoint. No package installation, model download, cloud job or paid API generation completed. New API spending is $0.

## Reproduce

From the vlm-xg-benchmark repository root, with NumPy/Pillow:

```bash
python experiments/statsbomb_finetune/prepare.py
python experiments/statsbomb_finetune/baselines.py
python -m unittest discover -s experiments/statsbomb_finetune -p 'test_*.py' -v
python experiments/statsbomb_finetune/train.py
```

In a separate compatible Python/CUDA training environment, install the dependencies described in experiments/statsbomb_finetune/README.md. Run a genuine one-batch forward/backward smoke test first, then the two full arms. Downloads are public pretrained weights, with no paid service calls:

```bash
python experiments/statsbomb_finetune/train.py --execute --arm lora --smoke --allow-download
python experiments/statsbomb_finetune/train.py --execute --arm head
python experiments/statsbomb_finetune/train.py --execute --arm lora
python experiments/statsbomb_finetune/evaluate.py
```

## Interpretation and provenance

These are reused source matches from earlier experiments. The new train/validation/test groups do not overlap, but the pilot is exploratory rather than untouched independent confirmation. Training on diagrams also does not establish transfer to real broadcast frames. The filtered pool excludes penalties, invalid/out-of-half shooters, missing/invalid freeze frames, unrecorded opposing keepers, unknown outcomes and missing reference xG; per-match exclusion counts are retained. Distances are StatsBomb normalized pitch-coordinate units, not verified physical metres. The image-feature logistic uses distance, opening angle and opponents in the shooting cone; the secondary header model has extra explicit body-part information.

Data source: StatsBomb Open Data, https://github.com/hudl/open-data. Retain StatsBomb attribution and follow the repository license. The new renderer differs from the original study renderer; older zero-shot scores are not inserted into this new comparison.

Files: protocol.json, match_plan.json, source_audit.json, selected.json, train.jsonl, validation.jsonl, test.jsonl, images/, training_preview.png, baseline_models.json, baseline_validation.json, training_preflight.json, audit.json.
