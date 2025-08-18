@echo off

cd %~dp0..\

REM cd D:/public/AI/framepack_cu126_torch26/webui

REM call environment.bat

set DIR=%~dp0..\..\system

set PATH=%DIR%\git\bin;%DIR%\python;%DIR%\python\Scripts;%PATH%
set PY_LIBS=%DIR%\python\Scripts\Lib;%DIR%\python\Scripts\Lib\site-packages
set PY_PIP=%DIR%\python\Scripts
set SKIP_VENV=1
set PIP_INSTALLER_LOCATION=%DIR%\python\get-pip.py



"%DIR%\python\python.exe" %~dp0..\run-sync.py --del_previous_file --shutdown --gpu_memory_preservation=12.0 --resolution=768 --total_second_length=10.0 -S=E:\tmp\image-to-video --fps=24 --prompt="Cute little girl, showing cute and lewd expressions, showing enchanting movements."

:done
pause