@echo off
setlocal

set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"

if "%~1"=="" (
	echo Usage: %~nx0 BUILD_DIR [cmake --build args...]
	exit /b 1
)

if not exist "%VSDEVCMD%" (
	echo Visual Studio developer environment not found:
	echo   %VSDEVCMD%
	exit /b 1
)

call "%VSDEVCMD%" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%

cmake --build %*
exit /b %errorlevel%
