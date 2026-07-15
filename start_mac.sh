#!/bin/zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 가상환경 활성화
source .venv/bin/activate

# Docker PostgreSQL 기동 확인
if ! docker compose ps | grep -q "healthy"; then
    echo "PostgreSQL 컨테이너를 시작합니다..."
    docker compose up -d
    echo "healthy 상태 대기 중..."
    until docker compose ps | grep -q "healthy"; do
        sleep 1
    done
fi

echo "Chat App 시작: http://localhost:8501"
streamlit run apps/chat_app.py --server.port 8501
