@echo off
:: 设置到全局环境变量
setx FRAMEPACK_ARGS "%~1"
:: 获得管理员权限
Net session >nul 2>&1 || mshta vbscript:CreateObject("Shell.Application").ShellExecute("cmd.exe","/c %~s0","","runas",1)(window.close)&&exit
:: 进入conda环境
:: CALL conda activate wan2.1

@REM 支持中文
chcp 65001
set PYTHONIOENCODING=utf-8
@REM 移动到项目根目录,更目录在当前目录下
cd /d %~dp0webui
@REM 输出当前目录
echo 当前运行目录: %CD%

SET ARGS=%FRAMEPACK_ARGS%
:: 删除环境变量
setx FRAMEPACK_ARGS ""
echo ARGS: %ARGS%

REM call environment.bat

set DIR=%~dp0system

set PATH=%DIR%\git\bin;%DIR%\python;%DIR%\python\Scripts;%PATH%
set PY_LIBS=%DIR%\python\Scripts\Lib;%DIR%\python\Scripts\Lib\site-packages
set PY_PIP=%DIR%\python\Scripts
set SKIP_VENV=1
set PIP_INSTALLER_LOCATION=%DIR%\python\get-pip.py



"%DIR%\python\python.exe" %~dp0webui\run-sync.py --max_run_time="7,1,1" --del_source_file --only_remain_last_file --shutdown --gpu_memory_preservation=12.0 --resolution=1024 --total_second_length=10.0 -S=E:\tmp\image-to-video\run --fps=20 --prompt="girl"

REM :done
pause
