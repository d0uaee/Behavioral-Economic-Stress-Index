@ECHO OFF

pushd %~dp0

IF "%SPHINXBUILD%" == "" (
    SET SPHINXBUILD=sphinx-build
)
SET SOURCEDIR=.
SET BUILDDIR=_build

%SPHINXBUILD% >NUL 2>NUL
IF ERRORLEVEL 9009 (
    ECHO.
    ECHO The 'sphinx-build' command was not found.
    ECHO Install the documentation dependencies with:
    ECHO   python -m pip install -r docs\requirements.txt
    ECHO.
    popd
    EXIT /B 1
)

IF "%1" == "" GOTO help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR%
GOTO end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR%

:end
popd

