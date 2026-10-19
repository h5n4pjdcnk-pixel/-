<#
  Neo4j 一键下载脚本（含 JDK）
  用法：
    powershell -ExecutionPolicy Bypass -File .\下载Neo4j.ps1 -Target desktop
    powershell -ExecutionPolicy Bypass -File .\下载Neo4j.ps1 -Target community
    powershell -ExecutionPolicy Bypass -File .\下载Neo4j.ps1 -Check        # 只测网络不下文件

  说明：Neo4j 官方安装包托管在 dist.neo4j.org（AWS CloudFront），该 CDN 按国家/地区
        屏蔽中国大陆，直连必然 403，必须走代理。本脚本会自动探测本机代理端口。
#>
param(
    [ValidateSet('desktop','community','both')][string]$Target = 'desktop',
    [string]$OutDir = "$env:USERPROFILE\Downloads\neo4j",
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$DesktopUrl   = 'https://neo4j.com/artifact.php?name=neo4j-desktop-offline-2.2.1-setup.exe'
$CommunityUrl = 'https://neo4j.com/artifact.php?name=neo4j-community-5.26.30-windows.zip'
$JdkUrl       = 'https://mirrors.huaweicloud.com/openjdk/17.0.2/openjdk-17.0.2_windows-x64_bin.zip'

# ---------- 1. 探测本机可用代理 ----------
function Get-ProxyCandidates {
    $ports = New-Object System.Collections.Generic.List[int]
    try {
        $ie = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction SilentlyContinue
        if ($ie.ProxyServer -match '(\d{2,5})\s*$') { $ports.Add([int]$Matches[1]) }
    } catch {}
    foreach ($p in 7890,7891,7897,7899,10808,10809,1080,2080,33210,20171) { $ports.Add($p) }

    $result = @()
    foreach ($p in ($ports | Sort-Object -Unique)) {
        $client = New-Object System.Net.Sockets.TcpClient
        try {
            $ok = $client.ConnectAsync('127.0.0.1', $p).Wait(400)
            if ($ok -and $client.Connected) { $result += "http://127.0.0.1:$p" }
        } catch {} finally { $client.Dispose() }
    }
    return $result
}

$proxyList = Get-ProxyCandidates
if ($proxyList.Count -gt 0) { Write-Host "探测到本机代理端口: $($proxyList -join ', ')" -ForegroundColor Green }
else { Write-Host "未探测到本机代理端口（FlClash 里需要打开“系统代理”或“TUN 模式”）" -ForegroundColor Yellow }

# ---------- 2. 判断哪条路能下 ----------
function Test-DownloadRoute {
    param([string]$TestUrl, [string]$TestProxy)
    $curlArgs = @('-sS', '-L', '-r', '0-0', '-o', 'NUL', '-w', '%{http_code}', '--max-time', '25', '-A', 'Mozilla/5.0')
    if ($TestProxy) { $curlArgs += @('-x', $TestProxy) }
    $curlArgs += $TestUrl
    $code = (& curl.exe @curlArgs 2>$null)
    return ("$code" -match '^(200|206)$')
}

$useProxy = $null
if (Test-DownloadRoute -TestUrl $DesktopUrl) {
    Write-Host "直连可用（少见）" -ForegroundColor Green
} else {
    foreach ($p in $proxyList) {
        if (Test-DownloadRoute -TestUrl $DesktopUrl -TestProxy $p) { $useProxy = $p; break }
    }
}

if (-not $useProxy) {
    Write-Host ""
    Write-Host "结论：直连和代理都下不到 Neo4j 安装包。" -ForegroundColor Red
    Write-Host "原因：dist.neo4j.org 是 AWS CloudFront 且按国家屏蔽大陆（403 Access Denied）。" -ForegroundColor Red
    Write-Host "请打开 FlClash -> 打开“系统代理”或“TUN 模式”，然后重新运行本脚本。" -ForegroundColor Yellow
    exit 2
}
Write-Host "下载通道: 代理 $useProxy" -ForegroundColor Green

if ($Check) { Write-Host "Check 模式：网络已通，未下载文件。"; exit 0 }

# ---------- 3. 下载 ----------
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

function Save-RemoteFile {
    param([string]$Url, [string]$FileName, [bool]$ViaProxy)
    $out = Join-Path $OutDir $FileName
    if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Force }
    Write-Host "`n开始下载: $FileName" -ForegroundColor Cyan
    $curlArgs = @('-sS', '-L', '--fail', '--retry', '2', '--retry-delay', '3')
    if ($ViaProxy) { $curlArgs += @('-x', $useProxy) }
    $curlArgs += @('-o', $out, $Url)
    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "下载失败: $FileName" -ForegroundColor Red; return $null }
    $len = (Get-Item -LiteralPath $out).Length
    Write-Host ("完成: {0}  ({1:N1} MB)" -f $FileName, ($len / 1MB)) -ForegroundColor Green
    return $out
}

$downloaded = @()
if ($Target -in @('desktop','both')) {
    $f = Save-RemoteFile -Url $DesktopUrl -FileName 'neo4j-desktop-2.2.1-setup.exe' -ViaProxy $true
    if ($f) { $downloaded += $f }
}
if ($Target -in @('community','both')) {
    $f = Save-RemoteFile -Url $CommunityUrl -FileName 'neo4j-community-5.26.30-windows.zip' -ViaProxy $true
    if ($f) { $downloaded += $f }
    $f = Save-RemoteFile -Url $JdkUrl -FileName 'openjdk-17.0.2_windows-x64_bin.zip' -ViaProxy $false
    if ($f) { $downloaded += $f }
}

# ---------- 4. 校验文件头 ----------
Write-Host "`n--- 文件校验 ---" -ForegroundColor Cyan
foreach ($f in $downloaded) {
    $bytes = [System.IO.File]::ReadAllBytes($f)[0..1]
    $magic = -join ($bytes | ForEach-Object { [char]$_ })
    $kind = switch ($magic) { 'MZ' { 'Windows 可执行程序(正确)' } 'PK' { 'ZIP 压缩包(正确)' } default { "魔数 '$magic' —— 可能下到的是错误页，需重下" } }
    Write-Host ("  {0}  {1:N1} MB  {2}" -f (Split-Path $f -Leaf), ((Get-Item -LiteralPath $f).Length / 1MB), $kind)
}

Write-Host "`n下载目录: $OutDir" -ForegroundColor Green
Write-Host "`n下一步：" -ForegroundColor Yellow
Write-Host "  [Desktop]   双击 neo4j-desktop-2.2.1-setup.exe 安装 -> 新建项目 -> Add Local DBMS(选 5.x) -> 设密码 -> Start"
Write-Host "              地址即 bolt://localhost:7687，用户名 neo4j，密码写入 .env 的 NEO4J_PASSWORD"
Write-Host "  [Community] 先装 JDK：解压 openjdk-17.0.2 到 $env:USERPROFILE\jdk17，把 $env:USERPROFILE\jdk17\bin 加进 PATH"
Write-Host "              解压 neo4j-community-5.26.30-windows.zip 到 $env:USERPROFILE\neo4j，然后执行 $env:USERPROFILE\neo4j\bin\neo4j.bat console"
Write-Host "              首次访问 http://localhost:7474，用 neo4j/neo4j 登录并改密码"