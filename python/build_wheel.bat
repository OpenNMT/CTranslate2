@echo off
setlocal

set SCRIPT_DIR=%~dp0
set LOGFILE=%SCRIPT_DIR%build_log.txt

echo [1] Initializing MSVC... > %LOGFILE%

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
  set VS_INSTALL_DIR=%%i
)

if not defined VS_INSTALL_DIR (
  echo ERROR: Visual Studio not found >> %LOGFILE%
  exit /b 1
)

call "%VS_INSTALL_DIR%\VC\Auxiliary\Build\vcvarsall.bat" x64 >> %LOGFILE% 2>&1
echo [2] MSVC done >> %LOGFILE%

set CTRANSLATE2_ROOT=%SCRIPT_DIR%..\build\install
echo [3] CTRANSLATE2_ROOT=%CTRANSLATE2_ROOT% >> %LOGFILE%

cd /d %SCRIPT_DIR%
echo [4] Running uv build... >> %LOGFILE%
uv build --wheel --no-build-isolation >> %LOGFILE% 2>&1
echo [5] Exit code: %ERRORLEVEL% >> %LOGFILE%

endlocal
