@echo off
REM Windows: 이 파일을 더블클릭하면 주제 입력창이 뜨고 영상이 자동으로 만들어집니다.
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate.bat || (echo 먼저 README의 설치를 해주세요 & pause & exit /b 1)
git pull --quiet 2>nul
pip install -q -r requirements.txt 2>nul
python -m if60 make --format cinematic
pause
