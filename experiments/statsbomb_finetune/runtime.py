"""Dependency-free runtime diagnostics; works before NumPy or PyTorch is installed."""
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/statsbomb_finetune_v1"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
PACKAGES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "PIL", "numpy")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def preflight():
    missing = [p for p in PACKAGES if not available(p)]
    result = {"model_id": MODEL_ID, "python_executable": sys.executable,
        "python_version": platform.python_version(), "platform": platform.system(),
        "virtual_environment": sys.prefix != sys.base_prefix,
        "missing_packages": missing,
        "missing_pip_packages": ["Pillow" if p == "PIL" else p for p in missing],
        "paid_api_calls": 0, "cloud_gpu_jobs": 0}
    if missing:
        return {**result, "status": "blocked_missing_training_runtime"}
    try:
        import torch
    except Exception as error:
        return {**result, "status": "blocked_torch_import", "error_type": type(error).__name__,
                "reason": str(error)[:1000]}
    result.update(torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                  cuda_available=torch.cuda.is_available())
    if not torch.cuda.is_available():
        return {**result, "status": "blocked_cuda_unavailable"}
    try:
        result.update(gpu=torch.cuda.get_device_name(0),
            gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            bf16_supported=torch.cuda.is_bf16_supported(),
            compute_capability=list(torch.cuda.get_device_capability(0)),
            compiled_cuda_arches=torch.cuda.get_arch_list())
        x = torch.ones(16, device="cuda")
        result["cuda_kernel_passed"] = float((x*x).sum().item()) == 16.
        del x
    except Exception as error:
        return {**result, "status": "blocked_cuda_kernel_unsupported",
                "error_type": type(error).__name__}
    return {**result, "status": "runtime_ready_weights_not_yet_checked"}
