# Run: powershell -File tests/test_prebuilt_swap.ps1   (simulates a locked prebuilt dir by holding a file handle open)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\tools\prebuilt_swap.ps1")
$root = Join-Path ([IO.Path]::GetTempPath()) "swaptest-$([guid]::NewGuid().ToString('N'))"
function New-Case($n) {
    $d = Join-Path $root $n; New-Item -ItemType Directory -Force "$d\a" | Out-Null; New-Item -ItemType Directory -Force "$d.new" | Out-Null
    Set-Content "$d\a\old.txt" "old"; Set-Content "$d.new\new.txt" "new"; $d
}
function Check($ok, $msg) { if (-not $ok) { throw "FAIL: $msg" }; Write-Host "ok: $msg" }
try {
    # 1. plain swap
    $d = New-Case c1
    Invoke-PrebuiltSwap -Dir $d -Tmp "$d.new" -Old "$d.old" -Tries 2 -DelayMs 10
    Check ((Test-Path "$d\new.txt") -and -not (Test-Path "$d.old") -and -not (Test-Path "$d.new")) "plain swap"
    # 2. permanently locked: fails with a clear message, nothing changed
    $d = New-Case c2
    $fs = [IO.File]::Open("$d\a\old.txt", 'Open', 'Read', 'None')
    try {
        $msg = $null
        try { Invoke-PrebuiltSwap -Dir $d -Tmp "$d.new" -Old "$d.old" -Tries 3 -DelayMs 20 } catch { $msg = $_.Exception.Message }
        Check ($msg -match "handle") "locked dir gives a clear error ($msg)"
        Check ((Test-Path "$d\a\old.txt") -and (Test-Path "$d.new\new.txt") -and -not (Test-Path "$d.old")) "locked dir: prebuilt intact, new kept"
    } finally { $fs.Dispose() }
    # 3. lock released mid-retry: succeeds
    $d = New-Case c3
    $fs = [IO.File]::Open("$d\a\old.txt", 'Open', 'Read', 'None')
    $t = New-Object Timers.Timer 300; $t.AutoReset = $false
    $null = Register-ObjectEvent $t Elapsed -Action { $fs.Dispose() } ; $t.Start()
    Invoke-PrebuiltSwap -Dir $d -Tmp "$d.new" -Old "$d.old" -Tries 6 -DelayMs 100
    Check (Test-Path "$d\new.txt") "transient lock: swap succeeds after retry"
    # 4. new build locked: rolled back, prebuilt not left missing
    $d = New-Case c4
    $fs = [IO.File]::Open("$d.new\new.txt", 'Open', 'Read', 'None')
    try {
        $msg = $null
        try { Invoke-PrebuiltSwap -Dir $d -Tmp "$d.new" -Old "$d.old" -Tries 3 -DelayMs 20 } catch { $msg = $_.Exception.Message }
        Check ($msg -match "Rolled back") "locked new build gives rollback error"
        Check ((Test-Path "$d\a\old.txt") -and -not (Test-Path "$d.old")) "locked new build: previous prebuilt restored"
    } finally { $fs.Dispose() }
    Write-Host "ALL PASS"
} finally { Remove-Item -Recurse -Force $root -ErrorAction SilentlyContinue }
