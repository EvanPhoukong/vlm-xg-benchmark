# Windows training setup

A native Windows Python 3.12.14 environment has been created at:

`C:\Users\ephou\AppData\Local\Temp\xg_research_windows_ft_py312`

Windows detects the NVIDIA GeForce RTX 5060 (8 GB). The CUDA PyTorch installation was attempted from this agent session, but DNS resolution failed for both `download.pytorch.org` and `pypi.org`. No CUDA PyTorch wheel is cached locally; the observed cached Torch wheel is CPU-only. No model training or paid API calls have run.

Run these native commands in a normal **Windows PowerShell** window:

```powershell
Set-Location -LiteralPath '\\wsl.localhost\Ubuntu-26.04\home\evan\xG_Research\vlm-xg-benchmark'
$xgPython = Join-Path $env:TEMP 'xg_research_windows_ft_py312\Scripts\python.exe'
& $xgPython -m pip install 'torch>=2.8.0' --index-url https://download.pytorch.org/whl/cu128
& $xgPython -m pip install -r .\experiments\statsbomb_finetune\requirements.txt
& $xgPython .\experiments\statsbomb_finetune\train.py --execute --arm lora --smoke --allow-download
```

These install the CUDA training stack into the prepared Windows environment, check the GPU, download the public Qwen model and run one forward/backward training batch. They do not start a cloud GPU job or use paid model APIs. Failed or existing smoke-run output is preserved. Stop if an installation command reports an error.

After installation, the Windows environment can be invoked directly by the agent. The saved runtime check is `results/statsbomb_finetune_v1/windows_preflight.json`. To run both full local training arms and the fixed evaluation:

```powershell
& $xgPython .\experiments\statsbomb_finetune\train.py --execute --arm head
& $xgPython .\experiments\statsbomb_finetune\train.py --execute --arm lora
& $xgPython .\experiments\statsbomb_finetune\evaluate.py
```

Each arm preserves the existing one-hour/three-epoch limit, match splits and validation-only checkpoint selection. The real GPU smoke test must establish memory fit before full training. This setup has not yet completed installation or GPU training.

`run_windows.ps1` is also retained as a reviewed automation source. Its syntax check passed, but launching it in this agent session was denied because it is unsigned. It is not a verified executable setup route on this machine; no execution policy was changed.

Installation logs and the explicit Windows environment record are retained with the experiment. The project and datasets can remain in their existing shared folder; the Python training process is native Windows.
