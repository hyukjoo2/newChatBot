# ChatSELMA

로컬에서 완전히 실행되는 멀티 에이전트 AI 챗봇.
Ollama LLM + LangGraph + PostgreSQL(pgvector) + Streamlit 기반.

---

## 주요 기능

- **문서 기반 Q&A** — PDF, DOCX, TXT, 이미지 등을 업로드해 로컬 지식베이스 구성. 하이브리드(BM25 + 벡터) 검색 + Corrective RAG
- **웹 검색** — Naver Search API로 실시간 웹 정보 조회
- **날씨 조회** — wttr.in 기반 현재 날씨 / 오늘·내일 예보
- **이미지 분석** — 비전 모델로 업로드 이미지 시각적 내용 설명
- **이메일 초안 작성** — 자연어 지시만으로 이메일 생성
- **문서 요약** — 업로드된 문서를 구조적으로 요약
- **다단계 작업** — 여러 단계가 필요한 복합 요청 자동 처리

### 에이전트 구성

| 에이전트 | 역할 |
|---------|------|
| `orchestrator_agent` | 질문 분석 → 실행 계획 수립 → 에이전트 라우팅 → 결과 평가 |
| `rag_agent` | 로컬 문서 검색 (Corrective RAG) |
| `web_search_agent` | Naver API 웹 검색 |
| `weather_agent` | 날씨 조회 |
| `reasoning_agent` | 일반 추론 / 코딩 / 수학 |
| `summary_agent` | 문서 요약 |
| `task_agent` | 다단계 작업 |
| `email_agent` | 이메일 초안 |
| `image_agent` | 이미지 시각 분석 |

---

## 사전 요구사항

| 항목 | 버전 / 비고 |
|------|------------|
| Python | 3.10 이상 |
| Docker Desktop | PostgreSQL + pgvector 컨테이너 실행 |
| [Ollama](https://ollama.com) | LLM 로컬 실행 |
| Tesseract OCR | 이미지 파일 인제스천 시 필요 |
| Naver Search API 키 | 웹 검색 기능 사용 시 필요 ([발급](https://developers.naver.com)) |

---

## 설치 순서

### 1. 저장소 클론

```bash
git clone <repo-url>
cd newChatBot
```

### 2. Ollama 설치 및 모델 다운로드

[ollama.com](https://ollama.com)에서 설치 후:

```bash
# 메인 LLM (권장)
ollama pull qwen3:8b

# 임베딩 모델 (필수)
ollama pull nomic-embed-text

# 비전 모델 (이미지 분석 기능 사용 시)
ollama pull moondream
```

> `qwen2.5:7b`, `llama3.1:8b` 등 다른 Ollama 호환 모델도 사용 가능하며 `.env`의 `OLLAMA_MODEL`로 변경한다.

### 3. Tesseract OCR 설치 (이미지 파일 업로드 시 필요)

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng
```

### 4. Python 가상환경 생성 및 패키지 설치

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5. 환경변수 설정

```bash
cp env_template .env
```

`.env` 파일을 열어 아래 항목을 수정한다:

```dotenv
# 사용할 LLM 모델명 (ollama pull 한 모델명과 일치해야 함)
OLLAMA_MODEL=qwen3:8b

# Naver Search API (웹 검색 기능 사용 시)
NAVER_CLIENT_ID=<your_client_id>
NAVER_CLIENT_SECRET=<your_client_secret>

# PostgreSQL 비밀번호 (임의로 설정)
POSTGRES_PASSWORD=your_password

# DATABASE_URL도 위 비밀번호와 일치하도록 수정
DATABASE_URL=postgresql+psycopg://assistant:your_password@localhost:5432/local_assistant
```

### 6. PostgreSQL 컨테이너 시작

```bash
docker compose up -d
```

컨테이너 최초 실행 시 `docker/postgres/init/` 의 SQL 스크립트가 자동 실행되어 스키마와 pgvector 익스텐션이 생성된다.

정상 기동 확인:

```bash
docker compose ps   # Status: healthy 확인
```

### 7. 앱 실행

```bash
# macOS/Linux
source .venv/bin/activate
streamlit run apps/chat_app.py --server.port 8501

# 또는 start_mac.sh 사용 (Docker 상태 자동 확인 포함)
bash start_mac.sh
```

브라우저에서 `http://localhost:8501` 접속.

---

## 디렉토리 구조 (주요)

```
newChatBot/
├── apps/
│   └── chat_app.py          # Streamlit UI
├── backend/
│   ├── chatbot/
│   │   ├── graph.py         # LangGraph 그래프
│   │   ├── nodes/           # 에이전트 노드
│   │   ├── tools.py         # search_documents / web_search / get_weather
│   │   └── prompts.py       # 시스템 프롬프트
│   ├── services/            # 비즈니스 로직
│   ├── database/            # DB 레포지토리
│   ├── documents/           # 문서 파서
│   └── rag/                 # 청킹 / 임베딩 / 검색
├── docker/
│   └── postgres/init/       # DB 초기화 SQL
├── docker-compose.yml
├── env_template             # 환경변수 템플릿
├── requirements.txt
└── start_mac.sh             # macOS 빠른 실행 스크립트
```

---

## 문제 해결

**Ollama 연결 오류**
- Ollama 앱이 실행 중인지 확인: `ollama list`
- `OLLAMA_BASE_URL`이 `http://localhost:11434`인지 확인

**PostgreSQL 연결 오류**
- `docker compose ps`로 컨테이너 상태 확인
- `.env`의 `POSTGRES_PASSWORD`와 `DATABASE_URL` 비밀번호가 일치하는지 확인

**웹 검색 결과 없음**
- `.env`의 `NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET` 확인
- Naver Developers에서 `검색` API 사용 등록 여부 확인

**이미지 OCR 오류**
- `tesseract --version`으로 설치 확인
- `kor` 언어 데이터 설치 여부 확인: `tesseract --list-langs`
