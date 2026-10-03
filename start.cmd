@echo off
chcp 936 >nul 2>&1
setlocal enabledelayedexpansion
title PDF Translation Studio
cd /d "%~dp0"

rem ---------------------------------------------------------------------
rem  已经在跑的话，直接开浏览器
rem ---------------------------------------------------------------------
netstat -ano | findstr /c:"LISTENING" | findstr ":8760 " >nul 2>&1
if not errorlevel 1 (
  echo.
  echo   服务已在运行，正在打开浏览器...
  start "" "http://127.0.0.1:8760/"
  timeout /t 2 >nul
  exit /b 0
)

echo.
echo   ============================================================
echo    PDF Translation Studio   (BabelDOC + DeepSeek)
echo   ============================================================
echo.

set "RTPY=%~dp0runtime\python\python.exe"

if not exist "!RTPY!" (
  echo   [错误] 缺少内置运行环境 runtime\python\python.exe
  echo          请把整个压缩包完整解压后再运行，不要只解出一部分文件。
  echo.
  pause
  exit /b 1
)

rem ---------------------------------------------------------------------
rem  首次运行：把翻译引擎装进内置解释器。
rem  引擎与依赖约 500 MB，只在这一步需要联网，之后每次启动都是秒开。
rem  中断了也没关系，重新双击本文件会接着装。
rem ---------------------------------------------------------------------
"!RTPY!" -c "import babeldoc" >nul 2>&1
if errorlevel 1 (
  echo   首次运行：正在安装翻译引擎，需要联网下载约 500 MB。
  echo   请耐心等待，只需这一次。
  echo.
  set "HF_ENDPOINT=https://hf-mirror.com"
  rem 阿里云 PyPI 镜像在本机实测只有 ~85 KB/s，引擎体积约 600 MB，装完要 1.5 小时以上，
  rem 看起来就像"一直在下载"。清华/中科大镜像在同一台机器上有 2.7-7.3 MB/s，换过去并加上超时与重试。
  set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"
  set "PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn mirrors.ustc.edu.cn"
  set "PIP_EXTRA_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple"
  set "PIP_TIMEOUT=20"
  set "PIP_RETRIES=3"
  "!RTPY!" -m pip install --upgrade pip --quiet --break-system-packages --disable-pip-version-check
  "!RTPY!" -m pip install --upgrade --break-system-packages BabelDOC pymupdf --disable-pip-version-check
  if errorlevel 1 (
    echo.
    echo   [错误] 引擎安装失败，多半是网络中断。
    echo          检查网络后重新双击本文件即可续装，已下载的部分不会白费。
    echo.
    pause
    exit /b 1
  )
  echo.
  echo   引擎安装完成。
  echo.
)

"!RTPY!" -c "import pymupdf" >nul 2>&1
if errorlevel 1 (
  echo   [注意] 缺少 PyMuPDF，版面填充、对照视图与排版体检会自动跳过。
  echo          补装： "!RTPY!" -m pip install pymupdf
  echo.
)

"!RTPY!" -c "import rapidocr_onnxruntime" >nul 2>&1
if errorlevel 1 (
  echo   正在安装 OCR 后端（扫描件翻译、以及自动重建文字层错序的页面要用它）...
  set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"
  set "PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn mirrors.ustc.edu.cn"
  set "PIP_TIMEOUT=20"
  set "PIP_RETRIES=3"
  "!RTPY!" -m pip install --upgrade --break-system-packages rapidocr-onnxruntime --disable-pip-version-check
  if errorlevel 1 (
    echo   [注意] OCR 后端安装失败：扫描件翻译与自动重建错序页会不可用。
    echo          补装： "!RTPY!" -m pip install rapidocr-onnxruntime
    echo.
  )
)

echo   运行环境 : !RTPY!
echo   界面     : http://127.0.0.1:8760/
echo.
echo   浏览器会自动打开；关闭此窗口即停止服务。
echo.

"!RTPY!" "%~dp0server.py" %*
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
  echo   [错误] 服务异常退出，代码 %RC%。
  pause
) else (
  echo   服务已停止。
  timeout /t 3 >nul
)
endlocal
