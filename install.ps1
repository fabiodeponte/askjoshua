# Ask Joshua installer for Windows: a private AI assistant that runs entirely on your computer.
#
#   irm https://ask.joshua.earth/install.ps1 | iex
#
# What it does, in order (nothing is installed outside your user folder, and nothing needs administrator rights):
#   1. checks your computer: processor, graphics card, memory, free disk
#   2. asks a few questions and picks the model that fits
#   3. installs Ollama (the engine that runs the models), Python (through uv), and Joshua
#   4. downloads the model
#   5. adds Joshua to the Start menu, and if you want, starts it when you sign in
# Run it again at any time to change your choices; it skips what is already there.
#
# Settings for unattended installs (used by Joshua's own tests): JOSHUA_YES=1 answers every question with its default,
# JOSHUA_MODEL picks the model, JOSHUA_URL the Joshua package to install.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # PowerShell's progress bar makes downloads many times slower
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$JoshuaUrl = if ($env:JOSHUA_URL) { $env:JOSHUA_URL } else { "https://ask.joshua.earth/joshua.tar.gz" }
$HomeDir = Join-Path $HOME "AskJoshua"
$App = Join-Path $HomeDir "joshua"
$Url = "http://127.0.0.1:7860"

function Say($t) { Write-Host $t }
function Step($t) { Write-Host ""; Write-Host "> $t" -ForegroundColor White }
function Ok($t) { Write-Host "  OK  $t" -ForegroundColor Green }
function Warn($t) { Write-Host "  !   $t" -ForegroundColor Yellow }
function Die($t) { Write-Host ""; Write-Host "X $t" -ForegroundColor Red; throw $t }
function Dim($t) { Write-Host "  $t" -ForegroundColor DarkGray }
function Ask($q, $default) {  # returns $true for yes
    if ($env:JOSHUA_YES) { return ($default -eq "y") }
    $hint = if ($default -eq "y") { "[Y/n]" } else { "[y/N]" }
    $a = Read-Host "  $q $hint"
    if (-not $a) { $a = $default }
    return ($a -match "^[YySs]")
}
function Write-Utf8($path, $text) { [IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false)) }   # no BOM: Python reads it as plain UTF-8

Write-Host @"

     JJJJJ  OOO   SSSS H   H U   U  AAA
       J   O   O S     H   H U   U A   A
       J   O   O  SSS  HHHHH U   U AAAAA
    J  J   O   O     S H   H U   U A   A
     JJ     OOO  SSSS  H   H  UUU  A   A
        your AI, on your computer, under your control
"@

# ---------- 1. the computer ----------
Step "Checking your computer"
if (-not [Environment]::Is64BitOperatingSystem) { Die "Joshua needs 64-bit Windows." }
$build = [Environment]::OSVersion.Version.Build
if ($build -lt 17763) { Die "Joshua needs Windows 10 (version 1809) or later." }
New-Item -ItemType Directory -Force -Path $HomeDir | Out-Null
$Free = [math]::Floor((Get-PSDrive -Name ($HomeDir.Substring(0, 1))).Free / 1GB)
$Ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$Chip = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()
$Vram = 0; $Gpu = ""
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    try {
        $line = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | Select-Object -First 1)
        if ($line) { $parts = $line.Split(","); $Gpu = $parts[0].Trim(); $Vram = [math]::Round([int]$parts[1].Trim() / 1024) }
    } catch {}
}
Ok "$Chip - $Ram GB memory - $Free GB free disk - Windows build $build"
if ($Gpu) { Ok "Graphics: $Gpu with $Vram GB" }
else { Warn "No NVIDIA graphics card found: models will run on the processor, which is much slower." }
$LocalNo = ""
if ($Ram -lt 8) { $LocalNo = "it has $Ram GB of memory, and local models need at least 8" }
if ($Free -lt 8) { $LocalNo = "it has only $Free GB of free disk, and local models need about 8" }
if ($Free -lt 3) { Die "Joshua needs at least 3 GB of free disk, even with cloud models only. You have $Free GB." }

# How fast a model writes depends mostly on memory bandwidth: every word means reading the whole model once
$Bw = 50
$GpuBw = if ($Vram -ge 16) { 700 } elseif ($Vram -ge 10) { 450 } else { 300 }
function Words($gb, $need) { $b = $Bw; if ($Vram -ge $need) { $b = $GpuBw }; return [math]::Floor($b * 35 * 75 / 10000 / $gb) }
$Speed4 = Words 3 4; $Speed8 = Words 6 7

# ---------- 2. questions and model choice ----------
Step "A few questions"
$CloudText = "Joshua runs here, and your chats, files and memory stay on this computer, but every message is answered by a large model on Ollama's servers. You need an Ollama account with a few dollars of credit."
$CloudOnly = $false; $Cloud = $false; $Think = $false
if ($LocalNo) {
    Say "  This computer can't run Joshua's models itself: $LocalNo."
    Dim "It can still use cloud models only. $CloudText"
    if (-not (Ask "Install Joshua with cloud models only?" "y")) { Die "Nothing was installed." }
    $CloudOnly = $true
} else {
    Say "  Where should Joshua's answers come from? Normally from models on this computer (private, free, works offline)."
    Dim "Or from cloud models only: $CloudText"
    $def = "n"; if ($Speed4 -lt 3 -and -not $Gpu) { $def = "y"; Dim "Recommended here: local models would be very slow on this computer." }
    if (Ask "Use cloud models only?" $def) { $CloudOnly = $true }
}
if ($CloudOnly) {
    $Cloud = $true; $Label = "Gemma 4 31B (cloud)"; $ModelGb = 0
} else {
    $Big = if ($Vram -ge 7) { 16 } else { 8 }
    if ($Big -lt 16) { $Size = "4b"; $Speed = $Speed4; Dim "On this computer Joshua uses the 4-billion-parameter model, writing about $Speed4 words a second." }
    else { $Size = "8b"; $Speed = $Speed8; Dim "Joshua uses the 8-billion-parameter model, the best fit for this computer: about $Speed8 words a second." }
    $Label = "qwen3-vl $Size"; $ModelGb = if ($Size -eq "8b") { 6 } else { 3 }
    $FastCandidates = @("qwen3-vl:$Size-instruct")
    $ThinkCandidates = @("qwen3-vl:$Size-thinking", "qwen3-vl:$Size")
    if ($env:JOSHUA_MODEL) { $FastCandidates = @($env:JOSHUA_MODEL); $ThinkCandidates = @($env:JOSHUA_MODEL); $Label = $env:JOSHUA_MODEL }
    Say ""
    Say "  Deep thinking: a second version of the model that reasons step by step before answering."
    Dim "Better on hard questions, slower. You switch it on per message. Another ~$ModelGb GB."
    $def = if ($Speed -lt 4) { "n" } else { "y" }
    if ($env:JOSHUA_MODEL) { $def = "n" }
    if (Ask "Install the deep-thinking version?" $def) { $Think = $true }
    Say ""
    Dim "Images and fine-tuning aren't available on Windows yet. Joshua can still look at images."
    Say ""
    Say "  Cloud models (optional): much bigger models that run on Ollama's servers instead of this computer."
    Dim "Your chats, files and memory stay stored here, but each message in a cloud chat is sent to Ollama. You choose per chat."
    $def = if ($Speed -lt 4 -and -not $env:JOSHUA_MODEL) { "y" } else { "n" }
    if (Ask "Connect Joshua to Ollama's cloud models?" $def) { $Cloud = $true }
}
$Need = 2 + $ModelGb; if ($Think) { $Need += $ModelGb }; if ($CloudOnly) { $Need = 1 }
if ($Free -lt ($Need + 3)) { Die "This needs about $Need GB of disk (plus a little room) and you have $Free GB free." }
Ok ("Plan: $Label" + $(if ($Think) { " + deep thinking" } else { "" }) + $(if ($Cloud -and -not $CloudOnly) { " + cloud models" } else { "" }) + " - about $Need GB")
if (-not (Ask "Go ahead?" "y")) { Die "Nothing was installed." }

# ---------- 3. Ollama, Python, Joshua ----------
Step "Ollama (runs the models)"
function Find-Ollama {
    $c = Get-Command ollama -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $p = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $p) { return $p }
    return $null
}
$Ollama = Find-Ollama
if ($Ollama) { Ok "already installed" }
else {
    $setup = Join-Path $env:TEMP "OllamaSetup.exe"
    Dim "Downloading Ollama..."
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $setup -UseBasicParsing
    # wait for the setup program only: Start-Process -Wait would also wait for the Ollama app it opens, which keeps running
    $p = Start-Process -FilePath $setup -ArgumentList "/VERYSILENT", "/NORESTART", "/SUPPRESSMSGBOXES" -PassThru
    $p.WaitForExit()
    Remove-Item $setup -ErrorAction SilentlyContinue
    $Ollama = Find-Ollama
    if (-not $Ollama) { Die "Ollama didn't install. Install it from https://ollama.com/download/windows, then run this again." }
    Ok "installed"
}
function Test-Up($u) { try { Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 3 | Out-Null; return $true } catch { return $false } }
if (-not (Test-Up "http://127.0.0.1:11434/api/version")) {
    Start-Process -FilePath $Ollama -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 60 -and -not (Test-Up "http://127.0.0.1:11434/api/version"); $i++) { Start-Sleep 1 }
}
if (-not (Test-Up "http://127.0.0.1:11434/api/version")) { Die "Ollama didn't start. Open Ollama from the Start menu once, then run this installer again." }
Ok "running"

Step "Python"
$Uv = Join-Path $HOME ".local\bin\uv.exe"
if (-not (Test-Path $Uv)) {
    $env:UV_NO_MODIFY_PATH = "1"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression | Out-Null
}
if (-not (Test-Path $Uv)) { Die "Couldn't install uv (the Python installer)." }
$Py = Join-Path $HomeDir "venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { & $Uv venv -q --seed --python 3.12 (Join-Path $HomeDir "venv") }
& $Uv pip install -q --python $Py pip
Ok ("Python " + (& $Py -c "import platform; print(platform.python_version())") + " in $HomeDir\venv")

Step "Joshua"
New-Item -ItemType Directory -Force -Path $App | Out-Null
$pkg = Join-Path $env:TEMP "joshua.tar.gz"
if (Test-Path $JoshuaUrl) { Copy-Item $JoshuaUrl $pkg -Force } else { Invoke-WebRequest -Uri $JoshuaUrl -OutFile $pkg -UseBasicParsing }
& tar.exe -xzf $pkg -C $App --strip-components=1
if ($LASTEXITCODE -ne 0) { Die "Couldn't unpack Joshua." }
Remove-Item $pkg -ErrorAction SilentlyContinue
& $Uv pip install -q --python $Py -r (Join-Path $App "requirements.txt")
if ($LASTEXITCODE -ne 0) { Die "Couldn't install Joshua's Python packages." }
Ok ("Joshua " + (Get-Content (Join-Path $App "VERSION")) + " installed in $App")

# ---------- 4. models ----------
function Pull-First($tags) {
    foreach ($t in $tags) {
        & $Ollama pull $t
        if ($LASTEXITCODE -eq 0) { return $t }
        Warn "$t is not available, trying the next one"
    }
    return $null
}
$Fast = ""; $Model = ""
if (-not $CloudOnly) {
    Step "Downloading the model (this is the long part)"
    $Fast = Pull-First $FastCandidates
    if (-not $Fast) { Die "Couldn't download the model. Check your connection and run again." }
    Ok $Fast
    $Model = $Fast
    if ($Think) {
        $m = Pull-First $ThinkCandidates
        if ($m) { $Model = $m; Ok $m } else { Warn "deep-thinking version not available; skipping" }
    }
}
if ($Cloud) {
    Step "Cloud models"
    Dim "Sign in to Ollama in the browser page that opens, and approve this computer. Come back here when it's done."
    & $Ollama signin
    if ($LASTEXITCODE -ne 0) { Warn "sign-in didn't finish; you can do it later from Joshua (Models page)" }
    $c = Pull-First @("gemma4:31b-cloud", "glm-5.3-flash:cloud", "gpt-oss:120b-cloud")
    if ($c) { Ok "$c added (nothing downloaded: it runs on Ollama's servers)" } else { Warn "couldn't add a cloud model now; add one later from Joshua's Models page" }
    if ($CloudOnly) { if (-not $c) { $c = "gemma4:31b-cloud" }; $Fast = $c; $Model = $c; $Label = $c }
}
Write-Utf8 (Join-Path $App "config.json") ('{"model": "' + $Model + '", "fast": "' + $Fast + '", "label": "' + $Label + '", "tune_base": "Qwen/Qwen3-4B-Instruct-2507", "tune_template": "qwen3:4b-instruct-2507-q4_K_M"}')

# ---------- 5. launcher ----------
Step "Joshua launcher"
$Pyw = Join-Path $HomeDir "venv\Scripts\pythonw.exe"
# start.ps1 starts Ollama and Joshua if they aren't running, then opens Joshua in a window of its own
Write-Utf8 (Join-Path $HomeDir "start.ps1") @"
`$ErrorActionPreference = 'SilentlyContinue'
function Up(`$u) { try { Invoke-WebRequest -Uri `$u -UseBasicParsing -TimeoutSec 2 | Out-Null; return `$true } catch { return `$false } }
if (-not (Up 'http://127.0.0.1:11434/api/version')) { Start-Process -FilePath '$Ollama' -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep 4 }
if (-not (Up '$Url')) {
  `$env:PYTHONUTF8 = '1'; `$env:JOSHUA_NO_OPEN = '1'
  Start-Process -FilePath '$Py' -ArgumentList 'web.py' -WorkingDirectory '$App' -WindowStyle Hidden -RedirectStandardOutput '$HomeDir\joshua.log' -RedirectStandardError '$HomeDir\joshua-errors.log'
  for (`$i = 0; `$i -lt 60 -and -not (Up '$Url'); `$i++) { Start-Sleep 1 }
}
if (`$args -contains '-NoOpen' -or `$env:JOSHUA_STAY_HIDDEN) { exit 0 }
`$edge = @("`${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe", "`$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe", "`$env:ProgramFiles\Google\Chrome\Application\chrome.exe", "`$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe") | Where-Object { Test-Path `$_ } | Select-Object -First 1
if (`$edge) { Start-Process -FilePath `$edge -ArgumentList '--app=$Url' } else { Start-Process '$Url' }
"@
Write-Utf8 (Join-Path $HomeDir "start.cmd") "@echo off`r`npowershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"%~dp0start.ps1`" %*`r`n"
Write-Utf8 (Join-Path $HomeDir "stop.ps1") @"
# Stops Joshua until you open it again
Get-CimInstance Win32_Process | Where-Object { `$_.CommandLine -like '*web.py*' -and `$_.ExecutablePath -like '$HomeDir*' } | ForEach-Object { Stop-Process -Id `$_.ProcessId -Force }
Write-Host 'Joshua stopped.'
"@
$StartMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Joshua.lnk"
$Startup = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup\Joshua.lnk"
Write-Utf8 (Join-Path $HomeDir "uninstall.ps1") @"
# Removes Joshua. Your chats are in $App\conversations: copy them first if you want to keep them.
`$a = Read-Host 'Remove Joshua, its chats and its Python? [y/N]'
if (`$a -notmatch '^[Yy]') { exit 0 }
& '$HomeDir\stop.ps1'
Remove-Item -Force '$StartMenu', '$Startup' -ErrorAction SilentlyContinue
Set-Location `$HOME; Remove-Item -Recurse -Force '$HomeDir'
Write-Host 'Done. Ollama itself is still installed; remove it from Settings > Apps if you do not need it.'
"@
function New-Shortcut($path, $arguments) {
    $s = (New-Object -ComObject WScript.Shell).CreateShortcut($path)
    $s.TargetPath = "powershell.exe"
    $s.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$HomeDir\start.ps1`" $arguments"
    $s.WorkingDirectory = $HomeDir
    $ico = Join-Path $App "icon.ico"
    if (Test-Path $ico) { $s.IconLocation = $ico }
    $s.Description = "Joshua, your AI on your computer"
    $s.Save()
}
New-Shortcut $StartMenu ""
Ok "Joshua is in the Start menu"
if (Ask "Start Joshua automatically when you sign in? (opens at once; scheduled tasks can run)" "y") {
    New-Shortcut $Startup "-NoOpen"
    Ok "Joshua starts when you sign in"
}

Step "Done"
# Counting installations: the first time an installation finishes on this computer, fetch one empty file from the site. Nothing is sent with it.
$counted = Join-Path $HomeDir ".counted"
if (-not (Test-Path $counted) -and -not $env:JOSHUA_YES) {
    try { Invoke-WebRequest -Uri "https://ask.joshua.earth/installed?done" -UseBasicParsing -TimeoutSec 5 | Out-Null } catch {}
    New-Item -ItemType File -Path $counted -Force | Out-Null
}
Say "  Open Joshua from the Start menu. It opens at $Url"
Dim "Everything lives in $HomeDir - stop it with stop.ps1 - remove it with uninstall.ps1"
if (Ask "Start Joshua now?" "y") { & (Join-Path $HomeDir "start.ps1") $(if ($env:JOSHUA_YES) { "-NoOpen" } else { "" }) }
