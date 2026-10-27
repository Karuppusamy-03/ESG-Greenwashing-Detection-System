param(
    [switch]$Live,
    [switch]$Local
)

$ErrorActionPreference = "Continue"
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Write-Host "FAIL: Project virtual-environment Python was not found: $Python" -ForegroundColor Red
    exit 1
}

$TestFiles = @(
    Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "tests") `
        -Filter "test*.py" -File -Recurse -ErrorAction SilentlyContinue
)
if ($TestFiles.Count -eq 0) {
    Write-Host "FAIL: No unittest files matching test*.py were found." -ForegroundColor Red
    exit 1
}

$PythonModules = @(
    "app.py",
    "automatic_collector.py",
    "web_search.py",
    "online_claim_extractor.py",
    "online_evidence_matcher.py",
    "contradiction_detector.py",
    "feature_builder.py",
    "risk_predictor.py",
    "risk_model.py",
    "explainability.py",
    "timeline_analyzer.py",
    "database.py"
)
$ModulePaths = @(
    foreach ($Module in $PythonModules) {
        Join-Path $ProjectRoot $Module
    }
)
foreach ($ModulePath in $ModulePaths) {
    if (-not (Test-Path -LiteralPath $ModulePath -PathType Leaf)) {
        Write-Host "FAIL: Required Python module was not found: $ModulePath" -ForegroundColor Red
        exit 1
    }
}

$Failures = 0
$OfflineTestsPassed = 0
$OfflineTestsFailed = 0
$LiveStatus = "SKIPPED (not requested)"
$LocalStatus = "SKIPPED (not requested)"
$LiveSettingWasPresent = Test-Path Env:ESG_LIVE_TESTS
$PreviousLiveSetting = $env:ESG_LIVE_TESTS

try {
    Remove-Item Env:ESG_LIVE_TESTS -ErrorAction SilentlyContinue
    Write-Host "Optional live integration checks: disabled for the offline suite"

    Push-Location $ProjectRoot
    try {
        Write-Host "`nRunning offline unittest suite..."
        $OfflineTestOutput = @(& $Python -m unittest discover -s tests -v 2>&1)
        $TestExitCode = $LASTEXITCODE
        $OfflineTestOutput | ForEach-Object { Write-Host "$_" }
        $OfflineTestText = $OfflineTestOutput -join "`n"
        $RunMatch = [regex]::Match($OfflineTestText, "Ran\s+(\d+)\s+tests?")
        $SkipMatch = [regex]::Match($OfflineTestText, "skipped=(\d+)")
        $FailureMatch = [regex]::Match($OfflineTestText, "failures=(\d+)")
        $ErrorMatch = [regex]::Match($OfflineTestText, "errors=(\d+)")
        $OfflineTestCount = 0
        $OfflineSkipped = 0
        $OfflineFailures = 0
        $OfflineErrors = 0
        if ($RunMatch.Success) {
            $OfflineTestCount = [int]$RunMatch.Groups[1].Value
        }
        if ($SkipMatch.Success) {
            $OfflineSkipped = [int]$SkipMatch.Groups[1].Value
        }
        if ($FailureMatch.Success) {
            $OfflineFailures = [int]$FailureMatch.Groups[1].Value
        }
        if ($ErrorMatch.Success) {
            $OfflineErrors = [int]$ErrorMatch.Groups[1].Value
        }
        $OfflineTestsFailed = $OfflineFailures + $OfflineErrors
        $OfflineTestsPassed = [Math]::Max(
            0,
            $OfflineTestCount - $OfflineSkipped - $OfflineTestsFailed
        )
        if ($TestExitCode -eq 0) {
            Write-Host "PASS: Offline tests: $OfflineTestsPassed passed, 0 failed, $OfflineSkipped skipped" -ForegroundColor Green
        }
        else {
            Write-Host "FAIL: Offline tests: $OfflineTestsPassed passed, $OfflineTestsFailed failed, $OfflineSkipped skipped (exit code $TestExitCode)" -ForegroundColor Red
            $Failures++
        }

        Write-Host "`nChecking Python syntax..."
        $SyntaxCheck = "import ast,sys; [ast.parse(open(path, encoding='utf-8-sig').read(), filename=path) for path in sys.argv[1:]]; print('Syntax OK for ' + str(len(sys.argv)-1) + ' modules')"
        & $Python -c $SyntaxCheck @ModulePaths
        $SyntaxExitCode = $LASTEXITCODE
        if ($SyntaxExitCode -eq 0) {
            foreach ($ModulePath in $ModulePaths) {
                Write-Host "PASS: $(Split-Path -Leaf $ModulePath)" -ForegroundColor Green
            }
            Write-Host "PASS: Python syntax checks" -ForegroundColor Green
        }
        else {
            Write-Host "FAIL: Python syntax checks (exit code $SyntaxExitCode)" -ForegroundColor Red
            $Failures++
        }

        Write-Host "`nWorkflow coverage: Company analysis, evidence, risk, explainability, timeline, and persistence are exercised by the offline unittest suite."

        if ($Live) {
            $env:ESG_LIVE_TESTS = "1"
            Write-Host "`nRunning explicitly enabled bounded live web workflow checks..." -ForegroundColor Yellow
            $LiveTestOutput = @(
                & $Python -m unittest discover `
                    -s tests `
                    -p test_live_workflows.py `
                    -v 2>&1
            )
            $LiveExitCode = $LASTEXITCODE
            $LiveTestOutput | ForEach-Object { Write-Host "$_" }
            $LiveTestText = $LiveTestOutput -join "`n"
            if ($LiveExitCode -ne 0) {
                $LiveStatus = "FAILED"
                Write-Host "FAIL: Optional live workflow test (exit code $LiveExitCode)" -ForegroundColor Red
                $Failures++
            }
            elseif ($LiveTestText -match "skipped=") {
                $LiveStatus = "SKIPPED (search results unavailable; see reported limitations)"
                Write-Host "SKIPPED: Optional live workflow test could not assess sources." -ForegroundColor Yellow
            }
            else {
                $LiveStatus = "PASSED"
                Write-Host "PASS: Optional live workflow test completed." -ForegroundColor Green
            }
        }
        else {
            Write-Host "Optional live workflow test: SKIPPED (not requested)"
        }

        if ($Local) {
            # Opt-in checks against this machine's private data/ and output/
            # folders. A missing file FAILS these checks; it is never passed.
            $env:ESG_LOCAL_DATA = "1"
            Write-Host "`nRunning opt-in local-data checks (requires your data/ and output/ folders)..." -ForegroundColor Yellow
            $LocalTestOutput = @(
                & $Python -m unittest discover `
                    -s local_checks `
                    -p "test_*.py" `
                    -v 2>&1
            )
            $LocalExitCode = $LASTEXITCODE
            $LocalTestOutput | ForEach-Object { Write-Host "$_" }
            if ($LocalExitCode -ne 0) {
                $LocalStatus = "FAILED (missing or incompatible local data)"
                Write-Host "FAIL: Local-data checks (exit code $LocalExitCode)" -ForegroundColor Red
                $Failures++
            }
            else {
                $LocalStatus = "PASSED"
                Write-Host "PASS: Local-data checks completed." -ForegroundColor Green
            }
        }
        else {
            Write-Host "Opt-in local-data checks: SKIPPED (not requested; use -Local)"
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($LiveSettingWasPresent) {
        $env:ESG_LIVE_TESTS = $PreviousLiveSetting
    }
    else {
        Remove-Item Env:ESG_LIVE_TESTS -ErrorAction SilentlyContinue
    }
}

if ($Failures -gt 0) {
    Write-Host "`nTEST SUMMARY: FAIL ($OfflineTestsPassed offline tests passed, $OfflineTestsFailed failed; live: $LiveStatus; local-data: $LocalStatus; $Failures required check(s) failed)" -ForegroundColor Red
    exit 1
}

Write-Host "`nTEST SUMMARY: PASS ($OfflineTestsPassed offline tests passed, $OfflineTestsFailed failed; live: $LiveStatus; local-data: $LocalStatus; syntax checks passed)" -ForegroundColor Green
exit 0
