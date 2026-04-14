@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"
set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"

if not exist "%VSDEVCMD%" (
	echo Visual Studio developer environment not found:
	echo   %VSDEVCMD%
	exit /b 1
)

call "%VSDEVCMD%" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%

cmake %*
if errorlevel 1 exit /b %errorlevel%

set "BUILD_DIR="
set "EXPECT_BUILD_DIR="

for %%A in (%*) do (
    if defined EXPECT_BUILD_DIR (
        set "BUILD_DIR=%%~A"
        set "EXPECT_BUILD_DIR="
    ) else (
        if "%%~A"=="-B" set "EXPECT_BUILD_DIR=1"
    )
)

if not defined BUILD_DIR exit /b 0
if not exist "%BUILD_DIR%\compile_commands.json" exit /b 0

echo Updating compile_commands.json...

if exist "%ROOT_DIR%\compile_commands.json" (
    echo Delete old compile_commands.json...
    del /f /q "%ROOT_DIR%\compile_commands.json"
)

copy /y "%BUILD_DIR%\compile_commands.json" "%ROOT_DIR%\compile_commands.json" >nul
if errorlevel 1 exit /b %errorlevel%

echo compile_commands.json updated.
