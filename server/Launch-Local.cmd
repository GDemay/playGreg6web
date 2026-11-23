@echo off
setlocal
start "" http://127.0.0.1:8000/
python "%~dp0serve_local.py" %*
