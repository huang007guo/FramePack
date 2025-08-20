@echo off
cd %~dp0webui
REM call environment.bat

set DIR=%~dp0system

set PATH=%DIR%\git\bin;%DIR%\python;%DIR%\python\Scripts;%PATH%
set PY_LIBS=%DIR%\python\Scripts\Lib;%DIR%\python\Scripts\Lib\site-packages
set PY_PIP=%DIR%\python\Scripts
set SKIP_VENV=1
set PIP_INSTALLER_LOCATION=%DIR%\python\get-pip.py



"%DIR%\python\python.exe" %~dp0webui\run-sync.py --max_run_time="7,1,1" --del_source_file --only_remain_last_file --shutdown --gpu_memory_preservation=12.0 --resolution=1024 --total_second_length=10.0 -S=E:\tmp\image-to-video\run --fps=20 --prompt="girl"

:done
pause