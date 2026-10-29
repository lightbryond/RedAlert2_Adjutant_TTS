@echo off
chcp 936 >nul
setlocal
title 心灵终结 副官语音合成服务

cd /d "%~dp0.."
set "PY=%~dp0..\venv\Scripts\python.exe"
set "PYTHONUNBUFFERED=1"

if not exist "%PY%" (
    echo.
    echo [错误] 找不到解释器：%PY%
    echo        请确认仓库根目录下有 venv 环境。
    echo.
    pause
    exit /b 1
)

echo ============================================================================
echo   心灵终结 · 四阵营副官语音合成服务
echo ============================================================================
echo.
echo   模型：visual\G_latest.pth
echo   配置：visual\modified_finetune_speaker.json
echo.
echo   启动后浏览器会自动打开 http://127.0.0.1:7860
echo   关闭本窗口即可停止服务（或按 Ctrl+C）
echo.
echo   如果 7860 被占用（例如上一次的服务没关干净），程序会自动改用
echo   7861、7862…… 并在窗口里显示实际地址，请以窗口输出为准。
echo.
echo   想换端口：      本脚本后加 --port 7861
echo   想局域网访问：  本脚本后加 --host 0.0.0.0
echo   想用公网链接：  本脚本后加 --share
echo   想强制用 CPU：  本脚本后加 --device cpu
echo.
echo ============================================================================
echo.

"%PY%" "%~dp0app.py" %*

echo.
echo ============================================================================
echo   服务已退出。
echo   若上面出现 "端口 xxx 已被占用"，说明已有一个服务在运行；
echo   可先关掉那个窗口，或在任务管理器里结束对应的 python.exe。
echo ============================================================================
pause
