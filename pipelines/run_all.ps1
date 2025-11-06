# ===============================
# Minimal MLB pipeline runner (robust ArgList, ASCII only)
# ===============================

param(
    [string]$Data = "out\mlb_features_prepared.csv",
    [string]$Python = "py",
    [string]$Env = ".venv",
    [switch]$Setup,
    [switch]$SkipEDA,
    [string]$Ingest = "",
    [string]$EDA = "",
    [string]$Features = "",
    [string]$Model = "",
    [switch]$ShowCmd
)

$ErrorActionPreference = 'Stop'

function RunPy([string]$ScriptPath, [string[]]$ArgList) {
    if (-not (Test-Path $ScriptPath)) { throw "Script not found: $ScriptPath" }

    # Build ONE safely quoted argument string: "<script>" "<arg1>" "<arg2>" ...
    $all     = @($ScriptPath) + $ArgList
    $quoted  = $all | ForEach-Object { '"' + ($_ -replace '"','\"') + '"' }
    $argLine = ($quoted -join ' ')
    if ($ShowCmd) { Write-Host ("CMD> {0} {1}" -f $Python, $argLine) -ForegroundColor DarkGray }

    $p = Start-Process -FilePath $Python -ArgumentList $argLine -NoNewWindow -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "Python failed (exit $($p.ExitCode)): $ScriptPath" }
}

function FindFirst([string[]]$Patterns) {
    foreach ($p in $Patterns) {
        $hit = Get-ChildItem -Recurse -Filter $p -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return ""
}

# Move to repo root (one level up from this script)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Resolve-Path (Join-Path $scriptDir '..'))

# Optional env setup
if ($Setup) {
    if (-not (Test-Path (Join-Path $Env 'Scripts\Activate.ps1'))) {
        & $Python -m venv $Env
    }
    . (Join-Path $Env 'Scripts\Activate.ps1')
    & $Python -m pip install --upgrade pip wheel setuptools
    if (Test-Path 'requirements.txt') { & $Python -m pip install -r requirements.txt }
    else { & $Python -m pip install numpy pandas scikit-learn matplotlib statsmodels pybaseball statsapi }
} else {
    $act = Join-Path $Env 'Scripts\Activate.ps1'
    if (Test-Path $act) { . $act }
}

# Auto-discover if not provided
if (-not $Ingest)   { $Ingest   = FindFirst @('data_ingestion.py','01_data_ingestion.py','scripts\data_ingestion.py') }
if (-not $EDA)      { $EDA      = FindFirst @('eda.py','02_eda.py','scripts\eda.py') }
if (-not $Features) { $Features = FindFirst @('feature_prep.py','03_feature_prep.py','scripts\feature_prep.py') }

# 1) Ingestion (optional)
if ($Ingest) { RunPy -ScriptPath $Ingest -ArgList @() }

# 2) EDA (optional)
if (-not $SkipEDA -and $EDA) { RunPy -ScriptPath $EDA -ArgList @() }

# 3) Feature preparation (optional; assumes script writes --out)
if ($Features) {
    $outDir = Split-Path $Data -Parent
    if ($outDir -and -not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
    RunPy -ScriptPath $Features -ArgList @('--out', $Data)
}

# Require prepared data
if (-not (Test-Path $Data)) { throw "Prepared data file not found: $Data" }

# 4) Model training (required)
if (-not $Model) { throw "Model script not found. Pass -Model '.\model_train_allinone.py'" }
RunPy -ScriptPath $Model -ArgList @(
    '--data', $Data,
    '--date-col', 'game_date',
    '--id-col', 'game_pk',
    '--target', 'home_win',
    '--outdir', 'model_all',
    '--topk', '25',
    '--threshold-strategy', 'f1',
    '--add-rolling',
    '--home-odds-col', 'home_moneyline',
    '--away-odds-col', 'away_moneyline',
    '--min-edge', '0.02',
    '--kelly-cap', '0.05'
)

Write-Host 'Pipeline finished successfully.'
