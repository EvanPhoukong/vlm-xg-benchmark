param(
    [switch]$Install,
    [switch]$Smoke,
    [switch]$TrainAll,
    [string]$BasePython
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).ProviderPath
$outputRoot = Join-Path $projectRoot 'results\statsbomb_finetune_v1'
$environmentPath = Join-Path ([System.IO.Path]::GetTempPath()) 'xg_research_windows_ft_py312'
$trainingPython = Join-Path $environmentPath 'Scripts\python.exe'
$environmentRecord = Join-Path $outputRoot 'windows_environment.json'

if (-not (Test-Path -LiteralPath $trainingPython)) {
    if (-not $BasePython -and (Test-Path -LiteralPath $environmentRecord)) {
        $BasePython = (Get-Content -LiteralPath $environmentRecord -Raw | ConvertFrom-Json).base_python
    }
    if (-not $BasePython -or -not (Test-Path -LiteralPath $BasePython)) {
        throw 'Provide -BasePython with the full path to an installed Windows Python 3.12 interpreter.'
    }
    & $BasePython -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) { throw 'Windows virtual environment creation failed.' }
}

function Invoke-TrainingPython {
    param([string[]]$PythonArguments)
    & $trainingPython @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Windows Python command failed with exit code $LASTEXITCODE. See the output above."
    }
}

Push-Location -LiteralPath $projectRoot
try {
    if ($Install) {
        Invoke-TrainingPython -PythonArguments @('-m','pip','install','torch>=2.8.0','--index-url','https://download.pytorch.org/whl/cu128','--disable-pip-version-check')
        Invoke-TrainingPython -PythonArguments @('-m','pip','install','-r',(Join-Path $PSScriptRoot 'requirements.txt'),'--disable-pip-version-check')
        Invoke-TrainingPython -PythonArguments @('-m','pip','check')
    }

    $statusFile = Join-Path $outputRoot 'windows_preflight.json'
    $trainer = Join-Path $PSScriptRoot 'train.py'
    Invoke-TrainingPython -PythonArguments @($trainer,'--status-file',$statusFile)
    $status = Get-Content -LiteralPath $statusFile -Raw | ConvertFrom-Json
    if ($status.status -ne 'runtime_ready_weights_not_yet_checked') {
        throw "Windows training preflight stopped: $($status.status). Use -Install to install the missing stack."
    }

    if ($Smoke) {
        Invoke-TrainingPython -PythonArguments @($trainer,'--execute','--arm','lora','--smoke','--allow-download')
    }
    if ($TrainAll) {
        Invoke-TrainingPython -PythonArguments @($trainer,'--execute','--arm','head','--allow-download')
        Invoke-TrainingPython -PythonArguments @($trainer,'--execute','--arm','lora')
        Invoke-TrainingPython -PythonArguments @((Join-Path $PSScriptRoot 'evaluate.py'))
        Invoke-TrainingPython -PythonArguments @((Join-Path $PSScriptRoot 'report.py'))
    }
} finally {
    Pop-Location
}
