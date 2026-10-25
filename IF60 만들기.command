#!/bin/bash
# Mac: 이 파일을 더블클릭하면 주제 입력창이 뜨고, 조사 → 대본 → 렌더링 → 재생까지 자동으로 진행됩니다.
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate 2>/dev/null || { echo "먼저 README의 설치를 해주세요 (.venv가 없어요)"; read -r; exit 1; }
git pull --quiet 2>/dev/null   # 최신 버전 받기 (실패해도 계속)
pip install -q -r requirements.txt 2>/dev/null   # 새로 필요한 부품이 생겼으면 설치
python -m if60 make --format cinematic
echo
read -r -p "창을 닫으려면 Enter를 누르세요..."
