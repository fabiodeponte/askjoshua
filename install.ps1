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
# The "Install Joshua" program (installer/windows/InstallJoshua.cs) runs this same script with JOSHUA_GUI=1 and shows it in a window.
# Settings for unattended installs (used by Joshua's own tests): JOSHUA_YES=1 answers every question with its default,
# JOSHUA_MODEL picks the model, JOSHUA_URL the Joshua package to install.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # PowerShell's progress bar makes downloads many times slower
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$JoshuaUrl = if ($env:JOSHUA_URL) { $env:JOSHUA_URL } else { "https://ask.joshua.earth/joshua.tar.gz" }
$HomeDir = Join-Path $HOME "AskJoshua"
$App = Join-Path $HomeDir "joshua"
$Url = "http://127.0.0.1:7860"

# How it prints. In a console: words wrapped at the window's edge, readable colours on PowerShell's blue, a few drawn symbols.
# In the "Install Joshua" window (JOSHUA_GUI): plain lines, which the window lays out itself.
$Gui = [bool]$env:JOSHUA_GUI
$Width = 100
if (-not $Gui) { try { $w = $Host.UI.RawUI.WindowSize.Width; if ($w -gt 30) { $Width = [math]::Min($w - 2, 110) } } catch {} }
function Wrap($text, $first, $rest) {  # the lines of $text, the first starting with $first, the others with $rest
    $lines = @(); $line = $first
    foreach ($word in ($text -split "\s+")) {
        if ($word -eq "") { continue }
        if ($line.Length + $word.Length + 1 -gt $Width -and $line.Trim().Length -gt 0) { $lines += $line.TrimEnd(); $line = $rest }
        if ($line -eq $first -or $line -eq $rest) { $line += $word } else { $line += " " + $word }
    }
    $lines += $line.TrimEnd(); return $lines
}
function Out($first, $rest, $text, $color) {
    if ($Gui) { Write-Host ($first + $text); return }
    foreach ($l in (Wrap $text $first $rest)) { Write-Host $l -ForegroundColor $color }
}
$Mark = if ($Gui) { ">" } else { [string][char]0x25B8 }   # the step marker the window looks for is ">"
function Say($t) { Out "  " "  " $t "White" }
function Dim($t) { Out "  " "  " $t "Gray" }
function Step($t) { Write-Host ""; Write-Host "$Mark $t" -ForegroundColor Cyan }
function Ok($t) { Out "  OK  " "      " $t "Green" }
function Warn($t) { Out "  !   " "      " $t "Yellow" }
function Die($t) { Write-Host ""; Out "X " "  " $t "Red"; throw $t }
function Ollama-SignIn {  # 'ollama signin' prints the page to open and waits until you approve this computer there
    $log = [IO.Path]::GetTempFileName(); $err = "$log.err"
    $p = Start-Process -FilePath $Ollama -ArgumentList "signin" -RedirectStandardOutput $log -RedirectStandardError $err -PassThru -WindowStyle Hidden
    $url = $null
    for ($i = 0; $i -lt 30 -and -not $url; $i++) {
        Start-Sleep -Milliseconds 500
        $text = ((Get-Content $log, $err -Raw -ErrorAction SilentlyContinue) -join "`n")
        if ($text -match 'https://ollama\.com\S+') { $url = $Matches[0] }
        if ($p.HasExited) { break }
    }
    ((Get-Content $log, $err -ErrorAction SilentlyContinue) | Where-Object { $_.Trim() }) | ForEach-Object { Dim $_.Trim() }
    if ($url) { Dim "Opening the page in your browser..."; Start-Process $url }
    $p.WaitForExit()
    Remove-Item $log, $err -ErrorAction SilentlyContinue
    return ($p.ExitCode -eq 0)
}

function Ask($q, $default) {  # returns $true for yes
    if ($env:JOSHUA_YES) { return ($default -eq "y") }
    if ($env:JOSHUA_GUI) {  # the "Install Joshua" window shows the question with Yes and No buttons, and writes back y or n
        [Console]::Out.WriteLine("@@ASK|$default|$q"); [Console]::Out.Flush()
        $a = [Console]::In.ReadLine()
        if (-not $a) { $a = $default }
        return ($a -match "^[YySs]")
    }
    $hint = if ($default -eq "y") { "[Y/n]" } else { "[y/N]" }
    Write-Host ""
    Write-Host "  $q $hint " -ForegroundColor Yellow -NoNewline
    $a = Read-Host
    if (-not $a) { $a = $default }
    return ($a -match "^[YySs]")
}
# The Ollama API key: you create it on ollama.com (no program can), copy it, and the installer takes it from the
# clipboard. It's checked with Ollama, saved only in providers.json, and the clipboard is emptied.
# It never appears on screen or in the log, and it isn't passed on a command line, where other programs could see it.
$SaveKey = @'
import json, pathlib, sys, requests
key = sys.stdin.read().strip()
if not key or len(key) > 300 or any(c.isspace() for c in key): sys.exit(2)
try:
    code = requests.get("https://ollama.com/api/balance", headers={"Authorization": "Bearer " + key}, timeout=15).status_code
except Exception:
    code = 0
if code in (401, 403): sys.exit(3)
p = pathlib.Path(sys.argv[1])
try: data = json.loads(p.read_text())
except Exception: data = {"keys": {}, "models": []}
data.setdefault("keys", {})["ollama"] = key
p.write_text(json.dumps(data, indent=1))
sys.exit(0 if code == 200 else 4)
'@
function Ollama-Key {
    Step "Ollama API key"
    Dim "Optional. With an API key, Joshua shows how much credit is left on your Ollama account, next to a cloud model's price. Cloud models work without it."
    if (-not (Ask "Set up the API key now?" "y")) { Dim "You can add it later in Joshua: Models page, under 'Open models in the cloud'."; return }
    Dim "A page opens on ollama.com: click 'Create API key', name it Joshua, then press the copy button next to the key."
    Dim "Don't paste it anywhere: when you answer Yes, the installer takes it from the clipboard, checks it with Ollama, saves it on this computer only, and empties the clipboard."
    Start-Process "https://ollama.com/settings/keys"
    $file = Join-Path $App "providers.json"
    for ($try = 0; $try -lt 2; $try++) {
        if (-not (Ask "Have you copied the key?" "y")) { break }
        $key = "" + (Get-Clipboard -Raw -ErrorAction SilentlyContinue)
        $key | & $Py -c $SaveKey $file
        $rc = $LASTEXITCODE; $key = $null
        if ($rc -eq 0 -or $rc -eq 4) {
            cmd /c "echo off | clip"   # empties the clipboard
            Ok "API key saved"
            if ($rc -eq 4) { Warn "Ollama didn't answer just now, so the key isn't checked yet: if the credit doesn't show in Joshua, paste a new one on the Models page" }
            return
        }
        if ($rc -eq 3) { Warn "Ollama refused that key. Copy it again (or make a new one) and answer Yes." }
        else { Warn "The clipboard doesn't hold an API key. Press the copy button next to the key and answer Yes." }
    }
    Dim "No key saved. You can add it later in Joshua: Models page, under 'Open models in the cloud'."
}
function Write-Utf8($path, $text) { [IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false)) }   # no BOM: Python reads it as plain UTF-8

# The name in big letters, drawn with block characters
$letters = @{
    J = @("  ###", "    #", "    #", "#   #", " ### "); O = @(" ### ", "#   #", "#   #", "#   #", " ### ")
    S = @(" ####", "#    ", " ### ", "    #", "#### "); H = @("#   #", "#   #", "#####", "#   #", "#   #")
    U = @("#   #", "#   #", "#   #", "#   #", " ### "); A = @(" ### ", "#   #", "#####", "#   #", "#   #")
}
$block = if ($Gui) { "#" } else { [string][char]0x2588 }
Write-Host ""
for ($row = 0; $row -lt 5; $row++) {
    $line = "  " + ((@("J", "O", "S", "H", "U", "A") | ForEach-Object { $letters[$_][$row].Replace("#", $block) }) -join "  ")
    Write-Host $line -ForegroundColor Green
}
Write-Host ""
Write-Host "  your AI, on your computer, under your control" -ForegroundColor Gray

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
    $Cloud = $true; $Label = "Nemotron 3 Super (cloud)"; $ModelGb = 0
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
    Dim "Your chats, files and memory stay stored here, but each message in a cloud chat is sent to Ollama. Joshua sets up Nemotron 3 Super (the one it starts with), DeepSeek V4.1 Flash (well under a cent per message) and Kimi K3.1 Pro (the most capable, and pricier). You choose per chat."
    $def = if ($Speed -lt 4 -and -not $env:JOSHUA_MODEL) { "y" } else { "n" }
    if (Ask "Connect Joshua to Ollama's cloud models?" $def) { $Cloud = $true }
}
$Account = $Cloud
if (-not $Cloud) {
    Say ""
    Say "  Ollama account (optional): you'll need one to use cloud models later. It's free to create."
    if (Ask "Sign in to Ollama, or create an account, now?" "n") { $Account = $true }
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
if (-not (Test-Path $Uv)) {  # uv's own release, unpacked here (its install script refuses to run where PowerShell scripts are forbidden)
    $arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "aarch64" } else { "x86_64" }
    $zip = Join-Path $env:TEMP "uv.zip"
    Invoke-WebRequest -Uri "https://github.com/astral-sh/uv/releases/latest/download/uv-$arch-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
    New-Item -ItemType Directory -Force -Path (Split-Path $Uv) | Out-Null
    Expand-Archive -Path $zip -DestinationPath (Split-Path $Uv) -Force
    Remove-Item $zip -ErrorAction SilentlyContinue
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
if ($Account) {
    Step "Ollama account"
    Dim "A page opens in your browser: sign in to Ollama, or choose 'Sign up' there to create a free account."
    Dim "Then approve this computer on that page, and come back here: the installation goes on by itself."
    if (Ollama-SignIn) { Ok "this computer is connected to your Ollama account" }
    else { Warn "sign-in didn't finish; you can do it later from Joshua (Models page)" }
}
if ($Cloud) {
    Step "Cloud models"
    $c = $null
    foreach ($m in @("nemotron-3-super:cloud", "deepseek-v4.1-flash:cloud", "kimi-k3.1-pro:cloud")) {  # Nemotron first: the cloud model Joshua starts with
        & $Ollama pull $m *> $null
        if ($LASTEXITCODE -eq 0) { Ok "$m added (nothing downloaded: it runs on Ollama's servers)"; if (-not $c) { $c = $m } }
        else { Warn "couldn't add $m now; add it later from Joshua's Models page" }
    }
    if ($CloudOnly) { if (-not $c) { $c = "nemotron-3-super:cloud" }; $Fast = $c; $Model = $c; $Label = $c }
}
Write-Utf8 (Join-Path $App "config.json") ('{"model": "' + $Model + '", "fast": "' + $Fast + '", "label": "' + $Label + '", "tune_base": "Qwen/Qwen3-4B-Instruct-2507", "tune_template": "qwen3:4b-instruct-2507-q4_K_M"}')
if ($Account) { Ollama-Key }

# ---------- 5. launcher ----------
Step "Joshua launcher"
$Pyw = Join-Path $HomeDir "venv\Scripts\pythonw.exe"
# Many companies forbid running .ps1 files ("execution policy"). Joshua's own scripts are therefore run as text, read from the file:
# that is allowed wherever PowerShell itself is.
function Run-Line($file, $extra) { return "-NoProfile -WindowStyle Hidden -Command `"& ([scriptblock]::Create([IO.File]::ReadAllText('$file'))) $extra`"" }
# start.ps1 starts Ollama and Joshua if they aren't running, then opens Joshua in a window of its own
Write-Utf8 (Join-Path $HomeDir "start.ps1") @"
`$ErrorActionPreference = 'SilentlyContinue'
function Up(`$u) { try { Invoke-WebRequest -Uri `$u -UseBasicParsing -TimeoutSec 2 | Out-Null; return `$true } catch { return `$false } }
if (-not (Up 'http://127.0.0.1:11434/api/version')) { Start-Process -FilePath '$Ollama' -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep 4 }
if (-not (Up '$Url')) {
  `$env:PYTHONUTF8 = '1'; `$env:JOSHUA_NO_OPEN = '1'; `$env:JOSHUA_LOG = '$HomeDir\joshua.log'
  Start-Process -FilePath '$Pyw' -ArgumentList 'web.py' -WorkingDirectory '$App'   # pythonw: Joshua runs without a window
  for (`$i = 0; `$i -lt 60 -and -not (Up '$Url'); `$i++) { Start-Sleep 1 }
}
if (`$args -contains '-NoOpen' -or `$env:JOSHUA_STAY_HIDDEN) { exit 0 }
`$edge = @("`${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe", "`$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe", "`$env:ProgramFiles\Google\Chrome\Application\chrome.exe", "`$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe") | Where-Object { Test-Path `$_ } | Select-Object -First 1
if (`$edge) { Start-Process -FilePath `$edge -ArgumentList '--app=$Url' } else { Start-Process '$Url' }
"@
Write-Utf8 (Join-Path $HomeDir "start.cmd") ("@echo off`r`npowershell " + (Run-Line "$HomeDir\start.ps1" "%*") + "`r`n")
Write-Utf8 (Join-Path $HomeDir "stop.ps1") @"
# Stops Joshua until you open it again
Get-CimInstance Win32_Process | Where-Object { `$_.CommandLine -like '*web.py*' -and `$_.ExecutablePath -like '$HomeDir*' } | ForEach-Object { Stop-Process -Id `$_.ProcessId -Force }
Write-Host 'Joshua stopped.'
"@
Write-Utf8 (Join-Path $HomeDir "stop.cmd") ("@echo off`r`npowershell " + (Run-Line "$HomeDir\stop.ps1" "") + "`r`n")
$StartMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Joshua.lnk"
$Startup = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup\Joshua.lnk"
Write-Utf8 (Join-Path $HomeDir "uninstall.ps1") @"
# Removes Joshua. Your chats are in $App\conversations: copy them first if you want to keep them.
`$a = Read-Host 'Remove Joshua, its chats and its Python? [y/N]'
if (`$a -notmatch '^[Yy]') { exit 0 }
& ([scriptblock]::Create([IO.File]::ReadAllText('$HomeDir\stop.ps1')))
Remove-Item -Force '$StartMenu', '$Startup' -ErrorAction SilentlyContinue
Set-Location `$HOME; Remove-Item -Recurse -Force '$HomeDir'
Write-Host 'Done. Ollama itself is still installed; remove it from Settings > Apps if you do not need it.'
"@
Write-Utf8 (Join-Path $HomeDir "uninstall.cmd") "@echo off`r`npowershell -NoProfile -Command `"& ([scriptblock]::Create([IO.File]::ReadAllText('$HomeDir\uninstall.ps1')))`"`r`npause`r`n"
function New-Shortcut($path, $arguments) {
    $s = (New-Object -ComObject WScript.Shell).CreateShortcut($path)
    $s.TargetPath = "powershell.exe"
    $s.Arguments = Run-Line "$HomeDir\start.ps1" $arguments
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
Dim "Everything lives in $HomeDir - stop it with stop.cmd - remove it with uninstall.cmd"
if (Ask "Start Joshua now?" "y") { & ([scriptblock]::Create([IO.File]::ReadAllText((Join-Path $HomeDir "start.ps1")))) $(if ($env:JOSHUA_YES) { "-NoOpen" } else { "" }) }
