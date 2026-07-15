source .venv/bin/activate

새 서버에 처음 설치하는 경우 전체 순서:
# 1. Docker Desktop 설치 (없는 경우)
# https://docs.docker.com/desktop/

# 2. 프로젝트 복사
git clone <repo> && cd newChatBot

# 3. .env 파일 생성 (없으면 DB 비밀번호 오류 발생)
cp .env.example .env   # 또는 직접 생성

# 4. PostgreSQL 컨테이너 실행
docker compose up -d

# 5. 정상 기동 확인
docker compose ps        # STATUS: healthy 확인
docker compose logs -f   # 로그 실시간 확인


Chat Application 실행
streamlit run apps/chat_app.py --server.port 8501
http://localhost:8501

Admin Application 실행 (문서 업로드 시)
별도 터미널에서:
streamlit run apps/admin_app.py --server.port 8502
http://localhost:8502


git

git add .
git commit -m "Initial commit"
git push -u origin main