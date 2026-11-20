# StatsBomb supervised image-to-xG pilot

Windows continuation: a native Windows Python 3.12 environment is prepared. Its CUDA PyTorch installation attempt failed because this agent session could not resolve the package download hosts. See `WINDOWS_SETUP.md` and `run_windows.ps1`; native Windows training is supported by the experiment code and does not require launching WSL. No GPU training has completed.

Prepared from 298 cached official match files: 2,000 training shots (181 goals, 207 matches), 400 validation shots (49 goals, 43 matches), and 500 test shots (63 goals, 48 matches). The eligible cache has 7,196 shots. All three splits are disjoint by whole match; selection is deterministic and does not balance goal labels. These source populations were used in earlier research, so this is an exploratory supervised experiment, not untouched confirmation.

**Current state:** Data preparation and statistical controls completed. The actual fine-tuning smoke launch stopped at preflight because the accessible Python has no PyTorch/Transformers/PEFT training stack. WSL execution is denied. No VLM was fine-tuned, no VLM validation/test result is claimed, and new API spending is $0.

The experiment compares Qwen2-VL-2B-Instruct with a frozen 4-bit backbone plus a linear probability head against the same model with text-attention QLoRA plus the same initial head. A sigmoid produces xG. Training uses unweighted BCE on actual goal/no-goal outcomes; outcomes are never fed into the image/prompt. The vision encoder stays frozen. No synthetic xG targets, teacher distillation, calibrator or probability averaging.

`prepare.py` builds a new deterministic 512×512 diagram renderer with a positional whitelist. The original source data, historical results and renderer are preserved. `baselines.py` fits an image-feature logistic reference (distance, angle, cone opponents), a separate header reference and the training goal-rate constant. StatsBomb's xG remains a separate comparator. Test evaluation stays locked until both VLM arms finish; checkpoint selection uses validation Brier only.

## Completed offline workflow

From the `vlm-xg-benchmark` repository root:

```bash
python experiments/statsbomb_finetune/prepare.py
python experiments/statsbomb_finetune/baselines.py
python -m unittest discover -s experiments/statsbomb_finetune -p 'test_*.py' -v
python experiments/statsbomb_finetune/train.py
python experiments/statsbomb_finetune/report.py
```

The runtime check (`train.py` without `--execute`) uses only Python's standard library and works even when NumPy is missing. It reports the Python executable, version, all missing packages and CUDA status. Preparation/analysis/data tests need NumPy and Pillow. No import, test or preparation call starts training or network access.

If Ubuntu's default `python` is a different environment, run the project's existing interpreter directly for this check:

```bash
.venv/bin/python experiments/statsbomb_finetune/train.py
```

That existing environment uses Python 3.14; the separate CUDA training environment proposed below uses Python 3.11 or 3.12.

## Local CUDA training

Use a separate Python 3.11 or 3.12 environment. Do not overwrite the existing project environment. The RTX 5060 needs a PyTorch wheel with Blackwell/CUDA support; the preflight runs an actual CUDA kernel to check compatibility. The following is a proposed setup, not a GPU-validated installation in this session:

```bash
python3.11 -m venv .venv-xg-finetune
source .venv-xg-finetune/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.8' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r experiments/statsbomb_finetune/requirements.txt
```

First run a real one-batch forward/backward check. It is labeled as a smoke test and produces no fitted-model accuracy claim:

```bash
python experiments/statsbomb_finetune/train.py --execute --arm lora --smoke --allow-download
```

The download flag permits fetching public pretrained weights into this experiment's `model_cache/`. Other invocations use cached weights only. No cloud GPU or paid API is invoked. An 8 GB GPU may still need memory adjustments; the genuine smoke test is required before full training. Changing training settings requires a new versioned protocol, not silently changing this study.

Then run both full arms and unlock the fixed test:

```bash
python experiments/statsbomb_finetune/train.py --execute --arm head
python experiments/statsbomb_finetune/train.py --execute --arm lora
python experiments/statsbomb_finetune/evaluate.py
```

Each arm has a one-hour wall-clock guard and at most three epochs. Failed runs and existing output directories are preserved; there is no automatic retry or implicit resume. Both arms must complete, use the same pretrained revision and have valid selected checkpoint hashes before test evaluation. Do not treat incomplete epochs or smoke outputs as completed models.

Results live in `results/statsbomb_finetune_v1/`: `REPORT.md`, `summary.json`, `protocol.json`, `match_plan.json`, `source_audit.json`, `selected.json`, JSONL split manifests, `images/`, `training_preview.png`, validation controls and runtime/audit records. New model checkpoints will appear only after real training.

Data source: StatsBomb Open Data, https://github.com/hudl/open-data. Retain StatsBomb attribution and follow the repository license. This is diagram training; it does not establish transfer to broadcast images.
