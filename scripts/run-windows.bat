@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1

rem =====================================================================
rem  Snippy - Windows starter.  Standalone: no Docker, no separate setup.
rem
rem    scripts\run-windows.bat             first run builds .venv, then starts
rem    scripts\run-windows.bat --check     validate only, do not connect
rem    scripts\run-windows.bat --asr       spoken triggers (installs the model)
rem
rem  Anything else you pass is forwarded straight to "python -m snippy".
rem  Output uses braille loading animations and ANSI colors. When the
rem  console cannot render them, every line falls back to plain text.
rem =====================================================================

pushd "%~dp0.."
if errorlevel 1 goto :no_root
set "ROOT=%CD%"
set "VENV=%ROOT%\.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "JOB=%TEMP%\snippy-job.cmd"
set "LOG=%TEMP%\snippy-job.log"
set "FLAG=%TEMP%\snippy-job.flag"

rem ------------------------------------------------------------- python ---
set "PY="
call python -c "import sys, operator; sys.exit(0 if operator.ge(sys.version_info[:2], (3, 11)) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=python"
if not defined PY (
    call py -3 -c "import sys, operator; sys.exit(0 if operator.ge(sys.version_info[:2], (3, 11)) else 1)" >nul 2>&1
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY goto :no_python

rem ------------------------------------------- colors and braille frames ---
rem cmd has no way to type an ESC character, so borrow one from a child
rem prompt (the pipe must hug $E or echo adds a trailing space). Then ask
rem the console to interpret ANSI escapes; if it refuses, the color
rem variables below stay undefined and every line prints as plain text.
rem stdout must NOT be redirected here: GetConsoleMode has to see the real
rem console handle (Windows isatty even claims the NUL device is a tty), so
rem only stderr is suppressed.
for /f "delims=" %%E in ('echo prompt $E^| cmd') do set "ESC=%%E"
call %PY% -c "import sys,ctypes,operator;k=ctypes.windll.kernel32;h=k.GetStdHandle(-11);m=ctypes.c_uint32();ok=k.GetConsoleMode(h,ctypes.byref(m)) and k.SetConsoleMode(h,operator.or_(m.value,4));sys.exit(0 if ok else 1)" 2>nul
if errorlevel 1 goto :no_colors
set "SNIPPY_COLORS=1"
set "C_RED=!ESC![31m"
set "C_YEL=!ESC![33m"
set "C_GRN=!ESC![32m"
set "C_CYN=!ESC![36m"
set "C_MAG=!ESC![35m"
set "C_DIM=!ESC![2m"
set "C_BLD=!ESC![1m"
set "C_OFF=!ESC![0m"
set "CLR=!ESC![2K!ESC![1G"
:no_colors
rem Ten braille spinner frames, one variable each: cmd substring slicing
rem works on bytes, so a single string cannot be indexed by character.
set "F0=⠋"
set "F1=⠙"
set "F2=⠹"
set "F3=⠸"
set "F4=⠼"
set "F5=⠴"
set "F6=⠦"
set "F7=⠧"
set "F8=⠇"
set "F9=⠏"
set "IDX=0"

rem ------------------------------------------------------------- banner ---
echo(
echo   !C_MAG!!C_BLD!Snippy!C_OFF!  !C_DIM!^|  %ROOT%!C_OFF!
echo   !C_DIM!⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿!C_OFF!
echo(
set "PYVER=unknown"
for /f "delims=" %%V in ('call %PY% -c "import sys;print(*sys.version_info[:3],sep=chr(46))"') do set "PYVER=%%V"
echo   !C_GRN!✓ ok!C_OFF!    python !PYVER!

rem --------------------------------------------------------------- venv ---
if exist "%VPY%" goto :venv_ready
set "SPIN_LABEL=creating the virtualenv in .venv"
set "SPIN_AFTER=venv_done"
> "%JOB%" (
    echo @echo off
    echo call %PY% -m venv "%VENV%" 1^> "%LOG%" 2^>^&1
    echo ^>^> "%FLAG%" echo %%ERRORLEVEL%%
)
goto :job_go
:venv_done
if "%RC%"=="0" goto :venv_ok
echo(
type "%LOG%"
goto :no_venv
:venv_ok
echo   !C_GRN!✓ ok!C_OFF!    virtualenv created in .venv
:venv_ready

rem -------------------------------------------------- forward arguments ---
set "WANT_ASR="
echo %* | findstr /i /c:"--asr" >nul
if not errorlevel 1 set "WANT_ASR=1"

rem ------------------------------------------- audioop on python 3.13+ ---
rem Python 3.13 removed the stdlib audioop module that discord.py imports,
rem so voice cannot work until the audioop-lts shim is installed.
set "NEED_AUDIOOP="
"%VPY%" -c "import sys, operator; sys.exit(0 if operator.ge(sys.version_info, (3, 13)) else 1)" >nul 2>&1
if not errorlevel 1 (
    "%VPY%" -c "import audioop" >nul 2>&1 || set "NEED_AUDIOOP=1"
)
if not defined NEED_AUDIOOP goto :audioop_ready
set "SPIN_LABEL=installing audioop support for python 3.13+"
set "SPIN_AFTER=audioop_done"
> "%JOB%" (
    echo @echo off
    echo "%VPY%" -m pip install --quiet --disable-pip-version-check audioop-lts 1^> "%LOG%" 2^>^&1
    echo ^>^> "%FLAG%" echo %%ERRORLEVEL%%
)
goto :job_go
:audioop_done
if "%RC%"=="0" goto :audioop_ok
echo   !C_YEL!▲ warn!C_OFF!  audioop-lts failed to install; voice may not work
goto :audioop_ready
:audioop_ok
echo   !C_GRN!✓ ok!C_OFF!    audioop support installed
:audioop_ready

rem ------------------------------------------------------ dependencies ---
rem Only reached when an import is actually broken, so a normal start is
rem instant and does not touch the network.
set "NEED_DEPS="
set "NEED_ASR="
"%VPY%" -c "import discord, discord.ext.voice_recv, av, numpy, PIL" >nul 2>&1 || set "NEED_DEPS=1"
if defined WANT_ASR (
    "%VPY%" -c "import faster_whisper" >nul 2>&1 || set "NEED_ASR=1"
)
if not defined NEED_DEPS if not defined NEED_ASR goto :deps_ready

set "REQ=%ROOT%\requirements.txt"
if defined NEED_ASR set "REQ=%ROOT%\requirements-asr.txt"
set "SPIN_LABEL=installing dependencies into .venv"
set "SPIN_AFTER=deps_done"
> "%JOB%" (
    echo @echo off
    echo "%VPY%" -m pip install --quiet --disable-pip-version-check -r "%REQ%" 1^> "%LOG%" 2^>^&1
    echo ^>^> "%FLAG%" echo %%ERRORLEVEL%%
)
goto :job_go
:deps_done
if "%RC%"=="0" goto :deps_ok
echo(
type "%LOG%"
goto :no_deps
:deps_ok
if defined NEED_ASR (
    echo   !C_GRN!✓ ok!C_OFF!    speech recognition installed
) else (
    echo   !C_GRN!✓ ok!C_OFF!    dependencies installed
)
:deps_ready

rem -------------------------------------------------------------- ffmpeg ---
where ffmpeg >nul 2>&1
if not errorlevel 1 goto :ffmpeg_ready
echo   !C_YEL!▲ warn!C_OFF!  ffmpeg was not found on PATH. Snippy needs it to encode clips.
echo           !C_DIM!Option A:  winget install --id Gyan.FFmpeg -e!C_OFF!
echo           !C_DIM!Option B:  https://www.gyan.dev/ffmpeg/builds/ then add its!C_OFF!
echo                       !C_DIM!bin folder to PATH!C_OFF!
set "GETFF="
set /p "GETFF=          Run the winget install now? [y/N] "
if /i "!GETFF!"=="y" (
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    echo(
    echo   !C_GRN!Installed.!C_OFF! Close this window and run this file again so the new
    echo   PATH is picked up.
    echo(
    pause
    exit /b 1
)
echo   !C_YEL!▲ warn!C_OFF!  continuing without ffmpeg; the check below will report it.
:ffmpeg_ready

rem ----------------------------------------------------------------- env ---
rem snippy reads .env itself on startup, so nothing here exports a token.
set "TOKEN="
if exist "%ROOT%\.env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%ROOT%\.env") do (
        if /i "%%A"=="DISCORD_TOKEN" set "TOKEN=%%B"
    )
)
if defined TOKEN if not "!TOKEN!"=="your-bot-token-here" goto :env_ready

echo(
echo   !C_MAG!Snippy needs a bot token.!C_OFF! Create one here, then paste it below:
echo     !C_DIM!https://discord.com/developers/applications!C_OFF!
echo   New Application - then Bot - Reset Token - copy it.
echo(
set "TOKEN="
set /p "TOKEN=  Bot token: "
if not defined TOKEN goto :no_token
if "!TOKEN!"=="your-bot-token-here" goto :no_token
set "GUILD="
set /p "GUILD=  Server ID, Enter to skip. Slash commands then appear instantly: "
> "%ROOT%\.env" (
    echo # Written by scripts\run-windows.bat. See .env.example for every option.
    echo DISCORD_TOKEN=!TOKEN!
    if defined GUILD echo DISCORD_GUILD_ID=!GUILD!
)
echo   !C_GRN!✓ ok!C_OFF!    .env written to %ROOT%\.env
:env_ready

rem ---------------------------------------------------------------- check ---
echo(
echo   !C_MAG!Verifying the setup!C_OFF!
echo(
"%VPY%" -m snippy --check %*
if not errorlevel 1 goto :check_ok
echo(
echo   !C_RED!✗ FAIL!C_OFF!  Snippy is not ready yet. Fix the FAIL lines above, then run this again.
echo(
pause
exit /b 1
:check_ok

echo %* | findstr /i /c:"--check" >nul
if not errorlevel 1 goto :check_only

rem ------------------------------------------------------------------ run ---
echo(
echo   !C_MAG!Starting Snippy!C_OFF!  !C_DIM!Press Ctrl+C to stop.!C_OFF!
if not defined SNIPPY_COLORS goto :run_skip_spin
set /a N=0
:launch_spin
call set "CH=%%F%IDX%%%"
set /a IDX+=1
if %IDX% geq 10 set /a IDX=0
<nul set /p "=!CLR!!C_CYN!!CH! !C_DIM!warming up the voice client!C_OFF!"
ping -n 1 127.0.0.1 >nul 2>&1
ping -n 1 127.0.0.1 >nul 2>&1
set /a N+=1
if %N% lss 7 goto :launch_spin
:run_skip_spin
<nul set /p "=!CLR!"
echo(
"%VPY%" -m snippy %*
set "RC=%ERRORLEVEL%"
echo(
if not "%RC%"=="0" echo   !C_RED!✗ fail!C_OFF!  Snippy exited with code !RC!.
echo(
pause
exit /b %RC%

rem ============================================ long-step job machinery ===
rem Steps that take a while (venv, pip) run as a background job that drops
rem its exit code into %FLAG%; while it works, a braille spinner animates
rem on a single line and the job log is kept out of the way. Without
rem colors the job runs in the foreground instead - same result, no spin.

:job_go
del /q "%FLAG%" >nul 2>&1
del /q "%LOG%" >nul 2>&1
set "TIMEOUT=0"
if defined SNIPPY_COLORS goto :job_bg
echo   ...   %SPIN_LABEL%
cmd /c "%JOB%"
goto :job_ran

:job_bg
start "" /b cmd /c "%JOB%"

:spin_flag
if exist "%FLAG%" goto :job_ran
set /a TIMEOUT+=1
if %TIMEOUT% gtr 5400 goto :spin_gave_up
call set "CH=%%F%IDX%%%"
set /a IDX+=1
if %IDX% geq 10 set /a IDX=0
<nul set /p "=!CLR!!C_CYN!!CH! !C_DIM!!SPIN_LABEL!!C_OFF!"
ping -n 1 127.0.0.1 >nul 2>&1
ping -n 1 127.0.0.1 >nul 2>&1
ping -n 1 127.0.0.1 >nul 2>&1
goto :spin_flag

:spin_gave_up
<nul set /p "=!CLR!"
set "RC=90091"
goto %SPIN_AFTER%

:job_ran
<nul set /p "=!CLR!"
rem The flag can exist for a moment before its content lands; retry a few
rem times so a half-written file is never read as a failure.
set "RETRY=0"
:read_flag
set "RC="
set /p RC=<"%FLAG%" 2>nul
if defined RC goto :flag_ok
set /a RETRY+=1
if %RETRY% gtr 300 goto :flag_dead
ping -n 1 127.0.0.1 >nul 2>&1
goto :read_flag
:flag_ok
del /q "%FLAG%" >nul 2>&1
goto %SPIN_AFTER%
:flag_dead
set "RC=90091"
goto %SPIN_AFTER%

rem ==================================================== failure branches ===
:check_only
echo(
echo   !C_GRN!✓  Check passed.!C_OFF! Not starting the bot.
echo(
pause
exit /b 0

:no_root
echo   !C_RED!✗ fail!C_OFF!  could not locate the Snippy folder from %~dp0
pause
exit /b 1

:no_python
echo(
echo   !C_RED!✗ fail!C_OFF!  Python 3.11 or newer was not found.
echo(
echo           Install it from !C_DIM!https://www.python.org/downloads/!C_OFF!
echo           and tick "Add python.exe to PATH", then run this file again.
echo(
pause
exit /b 1

:no_venv
echo(
echo   !C_RED!✗ fail!C_OFF!  could not create the virtualenv in %VENV%
echo           Make sure "python -m venv" works, then run this file again.
echo(
pause
exit /b 1

:no_deps
echo(
echo   !C_RED!✗ fail!C_OFF!  dependency install failed. Re-run this file to retry, or run:
echo           "%VPY%" -m pip install -r "%REQ%"
echo(
pause
exit /b 1

:no_token
echo(
echo   !C_RED!✗ fail!C_OFF!  a bot token is required. Set DISCORD_TOKEN in %ROOT%\.env
echo(
pause
exit /b 1
