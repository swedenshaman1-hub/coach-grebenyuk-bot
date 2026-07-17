@echo off
title NotebookLM Local Server (Grebenyuk)
echo Запускаю локальный NotebookLM сервер для бота Гребенюка...

set UV_PYTHON=C:\Users\Admin\AppData\Roaming\uv\tools\notebooklm-mcp-2026\Scripts\python.exe
set NOTEBOOKLM_LOCAL_SECRET=greb2026

cd /d "%~dp0"
"%UV_PYTHON%" nb_local_server.py
pause
