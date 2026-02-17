@echo off
REM Build HEC-DSS shared library for pyiwfm local development.
REM
REM Usage:
REM   build.bat              Build and install hecdss.dll to pyiwfm package
REM   build.bat --debug      Build debug configuration
REM   build.bat --no-install Build without installing to package
REM
REM Prerequisites:
REM   - CMake 3.16+
REM   - Visual Studio 2022 (or Ninja + compiler in PATH)
REM   - Python 3.10+
REM
REM The script auto-discovers heclib source from the IWFM build directory
REM (../src/build/_deps/heclib-src/) or fetches it from GitHub if not found.

setlocal

REM Set up Visual Studio environment if not already set
where cl >nul 2>&1
if errorlevel 1 (
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
    ) else (
        echo WARNING: Visual Studio environment not found. Ensure compiler is in PATH.
    )
)

REM Delegate to the Python build script
python "%~dp0build_hecdss.py" --install %*
