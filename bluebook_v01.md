# Local AI Assistant 프로젝트 설계

## 1. 프로젝트 목표

이 프로젝트의 목표는 `Ollama + Gemma 3 4B + LangChain + LangGraph + PostgreSQL + pgvector`를 사용해 로컬에서 동작하는 개인 업무 비서를 만드는 것이다.

시스템은 크게 두 개의 애플리케이션으로 구성한다.

1. Chat Application
2. Admin Application

Chat Application은 ChatGPT처럼 세션별 대화를 생성하고, 이전 대화를 조회하거나 삭제하며, 일반 대화와 RAG 기반 대화를 수행한다.

Admin Application은 업무 문서를 업로드하고, 텍스트 추출, OCR, 청크 생성, 임베딩 생성, 벡터 저장, 문서 삭제 및 재처리 등을 관리한다.

---

# 2. 핵심 기능

## 2.1 Chat Application

Chat Application은 일반 사용자가 사용하는 대화 화면이다.

주요 기능은 다음과 같다.

* 새 대화 세션 생성
* 세션 목록 조회
* 세션 선택
* 이전 대화 이어가기
* 세션 제목 변경
* 세션 삭제
* 전체 대화 기록 저장
* 일반 대화
* RAG 기반 대화
* 스트리밍 응답
* 답변 출처 및 페이지 표시

초기 버전에서는 다음 두 가지 대화 모드를 제공한다.

```text
일반 대화
지식 검색
```

추후 자동 모드를 추가할 수 있다.

```text
자동 모드
  ├─ 일반 질문 → 일반 대화
  └─ 업무 지식 질문 → RAG 검색
```

---

## 2.2 Admin Application

Admin Application은 개인 지식베이스를 관리하는 화면이다.

주요 기능은 다음과 같다.

* PDF 업로드
* TXT 업로드
* Markdown 업로드
* DOCX 업로드
* 이미지 업로드
* 원본 파일 저장
* 텍스트 추출
* OCR 처리
* 청크 생성
* 청크 미리보기
* 임베딩 생성
* pgvector 저장
* 문서 목록 조회
* 문서 상태 확인
* 문서 재처리
* 재벡터화
* 문서 비활성화
* 문서 삭제
* 태그 관리
* 카테고리 관리
* 오류 메시지 확인

Chat Application에서는 문서 업로드나 벡터 관리 기능을 제공하지 않는다.

Admin Application에서는 일반 채팅이나 대화 세션 관리 기능을 제공하지 않는다.

---

# 3. 최종 기술 스택

```text
생성 모델
  Ollama + gemma3:4b

임베딩 모델
  Ollama 기반 별도 임베딩 모델

LLM 연동
  LangChain

대화 및 RAG 워크플로
  LangGraph

데이터베이스
  PostgreSQL

벡터 저장 및 검색
  pgvector

프론트엔드
  Streamlit

원본 문서 저장
  로컬 파일 시스템

개발 언어
  Python
```

---

# 4. 전체 아키텍처

```text
┌───────────────────────────────┐
│ Chat Application              │
│                               │
│ - 세션 생성                   │
│ - 세션 조회                   │
│ - 세션 삭제                   │
│ - 일반 대화                   │
│ - RAG 대화                    │
│ - 대화 기록 조회              │
│ - 출처 표시                   │
└───────────────┬───────────────┘
                │
                │
┌───────────────▼───────────────┐
│ Common Backend                │
│                               │
│ - Session Service             │
│ - Chat Service                │
│ - Document Service            │
│ - Ingestion Service           │
│ - RAG Service                 │
│ - LangGraph                   │
│ - LangChain                   │
└───────┬───────────────┬───────┘
        │               │
        ▼               ▼
PostgreSQL           Ollama
+ pgvector           ├─ gemma3:4b
                     └─ embedding model
        │
        ▼
File System
원본 PDF / 이미지 / DOCX
```

Admin Application도 동일한 공통 Backend를 사용한다.

```text
┌───────────────────────────────┐
│ Admin Application             │
│                               │
│ - 문서 업로드                 │
│ - 텍스트 추출                 │
│ - OCR                         │
│ - 청크 미리보기               │
│ - 벡터화                      │
│ - 문서 재처리                 │
│ - 문서 삭제                   │
│ - 상태 관리                   │
│ - 태그 및 카테고리 관리       │
└───────────────┬───────────────┘
                │
                ▼
          Common Backend
```

---

# 5. PostgreSQL의 역할

PostgreSQL은 시스템의 기준 데이터 저장소다.

다음 데이터를 저장한다.

```text
사용자
대화 세션
전체 메시지 기록
문서 메타데이터
문서 청크
임베딩 벡터
문서 태그
문서 카테고리
문서 처리 상태
오류 정보
장기 사용자 기억
LangGraph 체크포인트
```

pgvector는 PostgreSQL 안에서 임베딩 벡터 검색을 담당한다.

별도의 Qdrant는 사용하지 않는다.

```text
PostgreSQL
  ├─ 관계형 데이터
  ├─ JSONB 메타데이터
  ├─ 전문 검색
  └─ pgvector 벡터 검색
```

---

# 6. 원본 파일 저장 방식

원본 문서는 PostgreSQL 내부에 저장하지 않는다.

파일 시스템에 저장하고 PostgreSQL에는 경로만 저장한다.

```text
data/
├── uploads/
│   └── {document_id}/
│       └── original.pdf
│
└── extracted/
    └── {document_id}/
        └── extracted.txt
```

예시:

```text
data/uploads/550e8400-e29b-41d4-a716-446655440000/original.pdf
```

PostgreSQL에는 다음 정보가 저장된다.

```text
document_id
file_name
original_path
extracted_text_path
mime_type
status
category
metadata
```

---

# 7. 핵심 데이터베이스 테이블

## 7.1 pgvector 확장

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 7.2 chat_sessions

대화 세션을 저장한다.

```sql
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 7.3 chat_messages

세션별 전체 대화 기록을 저장한다.

```sql
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_chat_messages_session
        FOREIGN KEY (session_id)
        REFERENCES chat_sessions(id)
        ON DELETE CASCADE
);
```

`role` 값 예시:

```text
user
assistant
system
tool
```

---

## 7.4 documents

업로드된 문서 정보를 저장한다.

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    file_name TEXT NOT NULL,
    original_path TEXT NOT NULL,
    extracted_text_path TEXT,
    mime_type TEXT,
    category TEXT,
    status TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 7.5 document_chunks

문서 청크와 임베딩 벡터를 저장한다.

```sql
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_document_chunks_document
        FOREIGN KEY (document_id)
        REFERENCES documents(id)
        ON DELETE CASCADE
);
```

`VECTOR(768)`은 예시다.

실제 차원은 사용하는 임베딩 모델의 차원과 반드시 같아야 한다.

예:

```text
768차원 모델 → VECTOR(768)
1024차원 모델 → VECTOR(1024)
```

---

## 7.6 document_tags

문서 태그를 저장한다.

```sql
CREATE TABLE document_tags (
    document_id UUID NOT NULL,
    tag TEXT NOT NULL,

    PRIMARY KEY (document_id, tag),

    CONSTRAINT fk_document_tags_document
        FOREIGN KEY (document_id)
        REFERENCES documents(id)
        ON DELETE CASCADE
);
```

---

## 7.7 ingestion_jobs

문서 처리 작업 상태를 저장한다.

```sql
CREATE TABLE ingestion_jobs (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ingestion_jobs_document
        FOREIGN KEY (document_id)
        REFERENCES documents(id)
        ON DELETE CASCADE
);
```

---

# 8. 문서 상태값

권장 상태값은 다음과 같다.

```text
UPLOADED
EXTRACTING
CHUNKING
EMBEDDING
READY
FAILED
INACTIVE
DELETED
```

처리 흐름은 다음과 같다.

```text
파일 업로드
   ↓
UPLOADED
   ↓
텍스트 추출 또는 OCR
   ↓
EXTRACTING
   ↓
청크 생성
   ↓
CHUNKING
   ↓
임베딩 생성
   ↓
EMBEDDING
   ↓
PostgreSQL + pgvector 저장
   ↓
READY
```

오류가 발생하면 다음과 같이 저장한다.

```text
status = FAILED
error_message = 오류 내용
```

---

# 9. 대화 세션과 LangGraph 연결

하나의 Chat 세션은 하나의 LangGraph thread와 대응한다.

```text
chat_sessions.id
        =
LangGraph thread_id
```

LangGraph 호출 예시는 다음과 같다.

```python
config = {
    "configurable": {
        "thread_id": str(session_id),
    }
}
```

현재 개발 단계에서 사용하는 `InMemorySaver`는 프로그램이 종료되면 대화 상태가 사라진다.

최종 구조에서는 PostgreSQL 기반 LangGraph checkpointer를 사용한다.

역할은 다음처럼 구분한다.

```text
chat_sessions
  세션 목록과 제목 관리

chat_messages
  전체 대화 기록 저장 및 UI 표시

LangGraph Checkpointer
  그래프 실행 상태와 대화 문맥 저장
```

---

# 10. Chat 처리 흐름

## 10.1 일반 대화 모드

```text
사용자 질문
   ↓
세션의 최근 대화 기록 조회
   ↓
시스템 프롬프트 구성
   ↓
gemma3:4b 호출
   ↓
응답 스트리밍
   ↓
사용자 메시지 저장
   ↓
AI 메시지 저장
```

---

## 10.2 RAG 대화 모드

```text
사용자 질문
   ↓
질문 임베딩 생성
   ↓
pgvector 유사도 검색
   ↓
관련 문서 청크 조회
   ↓
시스템 프롬프트 구성
   ↓
최근 대화 + 검색 문서 + 사용자 질문 전달
   ↓
gemma3:4b 답변 생성
   ↓
답변 및 출처 표시
   ↓
메시지와 출처 정보 저장
```

---

# 11. pgvector 검색 예시

```sql
SELECT
    id,
    document_id,
    page_number,
    content,
    metadata,
    1 - (embedding <=> :query_embedding) AS similarity
FROM document_chunks
ORDER BY embedding <=> :query_embedding
LIMIT 5;
```

`<=>`는 cosine distance 검색에 사용할 수 있다.

검색 결과는 다음 정보를 포함한다.

```text
document_id
file_name
page_number
chunk_index
content
similarity
```

---

# 12. RAG 답변 출처

RAG 답변에는 반드시 출처 정보를 표시한다.

예시:

```text
출처

1. SAC_Data_Action_Manual.pdf, 14페이지
2. Job_Monitor_Architecture.pdf, 8페이지
```

메시지 metadata에 출처를 저장할 수 있다.

```json
{
  "mode": "rag",
  "sources": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "file_name": "SAC_Data_Action_Manual.pdf",
      "page_number": 14,
      "similarity": 0.87
    }
  ]
}
```

---

# 13. LangGraph 초기 구조

초기 LangGraph는 복잡하게 만들지 않는다.

```text
START
  ↓
mode 확인
  ├─ chat → 일반 대화 생성
  └─ rag  → 문서 검색 → RAG 답변 생성
  ↓
END
```

개념적인 흐름:

```python
builder.add_node("route", route_node)
builder.add_node("chat", chat_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate_rag", generate_rag_node)

builder.add_edge(START, "route")

builder.add_conditional_edges(
    "route",
    select_mode,
    {
        "chat": "chat",
        "rag": "retrieve",
    },
)

builder.add_edge("retrieve", "generate_rag")
builder.add_edge("chat", END)
builder.add_edge("generate_rag", END)
```

추후 다음 기능을 추가한다.

```text
질문 자동 분류
검색어 재작성
검색 결과 평가
재검색
답변 검증
출처 검증
대화 요약
장기 사용자 기억
```

---

# 14. 긴 대화 처리

전체 대화 기록은 PostgreSQL에 모두 저장한다.

하지만 모든 메시지를 매번 LLM에 전달하지 않는다.

```text
PostgreSQL에 저장
  전체 대화 기록

LLM에 전달
  시스템 프롬프트
  이전 대화 요약
  최근 메시지
  검색 문서
  현재 질문
```

초기 기준은 다음과 같이 설정한다.

```text
최근 6~10개 메시지
+
이전 대화 요약
+
RAG 검색 문서
+
현재 질문
```

이전 대화가 길어지면 요약 데이터를 별도로 저장한다.

예:

```sql
ALTER TABLE chat_sessions
ADD COLUMN summary TEXT;
```

---

# 15. 개인 비서의 기억 구조

시스템은 세 가지 기억 구조를 가진다.

## 15.1 세션 기억

현재 대화 안에서 기억한다.

```text
LangGraph Checkpointer
```

## 15.2 전체 대화 기록

ChatGPT처럼 세션별 대화를 보관한다.

```text
chat_sessions
chat_messages
```

## 15.3 업무 지식과 장기 기억

업로드한 문서와 사용자의 장기 정보를 저장한다.

```text
documents
document_chunks
personal_memories
```

최종적으로는 다음과 같은 구조가 된다.

```text
최근 대화 기억
  LangGraph

전체 대화 기록
  PostgreSQL

업무 지식
  PostgreSQL + pgvector

장기 사용자 기억
  PostgreSQL
```

---

# 16. 문서 삭제 정책

문서를 삭제할 때 다음 항목을 모두 처리해야 한다.

```text
1. document_chunks 삭제
2. documents 삭제
3. document_tags 삭제
4. ingestion_jobs 삭제
5. 원본 파일 삭제
6. 추출 텍스트 삭제
```

외래키에 `ON DELETE CASCADE`를 사용하면 PostgreSQL 내부 데이터는 함께 삭제할 수 있다.

파일 시스템의 원본 파일과 추출 파일은 애플리케이션에서 직접 삭제해야 한다.

안전성을 위해 바로 삭제하기보다 비활성화 기능을 먼저 제공할 수 있다.

```text
ACTIVE
INACTIVE
DELETED
```

`INACTIVE` 문서는 검색 대상에서 제외한다.

---

# 17. 권장 프로젝트 구조

```text
newChatBot/
├── .env
├── .gitignore
├── requirements.txt
├── docker-compose.yml
├── README.md
│
├── apps/
│   ├── chat_app.py
│   └── admin_app.py
│
├── backend/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── chatbot/
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── routing.py
│   │   ├── prompts.py
│   │   └── nodes/
│   │       ├── __init__.py
│   │       ├── chat.py
│   │       ├── retrieve.py
│   │       └── generate.py
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   ├── models.py
│   │   ├── schema.py
│   │   └── repositories/
│   │       ├── __init__.py
│   │       ├── session_repository.py
│   │       ├── message_repository.py
│   │       ├── document_repository.py
│   │       └── chunk_repository.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chat_service.py
│   │   ├── session_service.py
│   │   ├── document_service.py
│   │   └── ingestion_service.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── chunker.py
│   │   └── ingestion.py
│   │
│   └── documents/
│       ├── __init__.py
│       ├── loaders.py
│       ├── pdf_loader.py
│       ├── image_loader.py
│       └── ocr.py
│
├── data/
│   ├── uploads/
│   └── extracted/
│
└── tests/
    ├── test_sessions.py
    ├── test_documents.py
    ├── test_rag.py
    └── test_vector_search.py
```

---

# 18. Streamlit 실행 구조

## Chat Application

```bash
streamlit run apps/chat_app.py --server.port 8501
```

접속 주소:

```text
http://localhost:8501
```

## Admin Application

```bash
streamlit run apps/admin_app.py --server.port 8502
```

접속 주소:

```text
http://localhost:8502
```

---

# 19. Chat Application 화면 구성

```text
┌─────────────────┬────────────────────────────────┐
│ + 새 대화       │ 세션 제목                      │
│                 │                                │
│ 오늘            │ 사용자 메시지                  │
│ - 세션 A        │ AI 응답                        │
│ - 세션 B        │                                │
│                 │ 출처                           │
│ 이전            │ - 문서명, 페이지               │
│ - 세션 C        │                                │
│                 │ [메시지 입력________________]  │
└─────────────────┴────────────────────────────────┘
```

왼쪽 사이드바 기능:

```text
새 대화
세션 목록
세션 선택
세션 제목 수정
세션 삭제
대화 모드 선택
```

메인 화면 기능:

```text
메시지 히스토리
스트리밍 응답
출처 표시
메시지 입력
```

---

# 20. Admin Application 화면 구성

```text
┌──────────────────────────────────────────────────┐
│ Knowledge Base Admin                             │
├──────────────────────────────────────────────────┤
│ [문서 업로드] [문서 목록] [청크 관리] [설정]    │
├──────────────────────────────────────────────────┤
│ 파일 선택                                        │
│ 카테고리                                         │
│ 태그                                             │
│ 처리 방식                                        │
│ 청크 크기                                        │
│ 오버랩                                           │
│                                                  │
│ [업로드 및 처리]                                 │
├──────────────────────────────────────────────────┤
│ 문서 목록                                        │
│                                                  │
│ SAC Manual.pdf    READY       328 chunks          │
│ Job Monitor.pdf   FAILED      재처리 / 삭제       │
└──────────────────────────────────────────────────┘
```

초기 청크 설정 예시:

```text
chunk size = 300 tokens
overlap = 80 tokens
```

문서 특성에 따라 관리 화면에서 변경할 수 있도록 한다.

---

# 21. 환경변수 예시

`.env`

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

LLM_TEMPERATURE=0.2
LLM_NUM_CTX=8192
LLM_NUM_PREDICT=1024

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=local_assistant
POSTGRES_USER=assistant
POSTGRES_PASSWORD=assistant_password

DATABASE_URL=postgresql+psycopg://assistant:assistant_password@localhost:5432/local_assistant

UPLOAD_DIR=data/uploads
EXTRACTED_DIR=data/extracted

DEFAULT_CHUNK_SIZE=300
DEFAULT_CHUNK_OVERLAP=80
DEFAULT_RETRIEVAL_TOP_K=5
```

---

# 22. 구현 순서

## 1단계: PostgreSQL 기반 대화 시스템

구현 항목:

```text
PostgreSQL + pgvector 실행
DB 연결
chat_sessions 생성
chat_messages 생성
세션 생성
세션 목록 조회
세션 선택
세션 삭제
메시지 저장
메시지 조회
LangGraph PostgreSQL Checkpointer 적용
Chat Application 구현
```

완료 시 제공되는 기능:

```text
새 대화
세션 선택
이전 대화 재개
세션 삭제
대화 기록 유지
일반 대화
```

---

## 2단계: Admin 문서 관리

구현 항목:

```text
문서 업로드
원본 저장
documents 테이블 저장
문서 목록 조회
문서 상세 조회
문서 삭제
문서 비활성화
처리 상태 표시
```

---

## 3단계: 문서 처리와 벡터화

구현 항목:

```text
PDF 텍스트 추출
OCR
청크 생성
청크 미리보기
임베딩 생성
pgvector 저장
문서 재처리
재벡터화
```

---

## 4단계: RAG 연결

구현 항목:

```text
질문 임베딩
pgvector 검색
관련 청크 조회
Gemma에 문서 전달
근거 기반 답변
출처 표시
메시지 metadata 저장
```

---

## 5단계: 고도화

구현 항목:

```text
질문 자동 분류
검색어 재작성
검색 결과 평가
재검색
답변 검증
긴 대화 요약
장기 사용자 기억
문서 태그 필터
카테고리 필터
하이브리드 검색
사용자 인증
관리자 인증
```

---

# 23. 최종 설계 결정

```text
애플리케이션
  Chat Application
  Admin Application

프론트엔드
  Streamlit

백엔드
  공통 Python Backend

생성 모델
  Ollama gemma3:4b

임베딩 모델
  Ollama 기반 별도 임베딩 모델

LLM 프레임워크
  LangChain

워크플로 및 상태 관리
  LangGraph

데이터베이스
  PostgreSQL

벡터 저장 및 검색
  pgvector

원본 문서
  로컬 파일 시스템
```

---

# 24. 가장 먼저 구현할 범위

첫 번째 구현 목표는 다음과 같다.

```text
PostgreSQL + pgvector 환경 구성
세션 테이블 생성
메시지 테이블 생성
세션 생성/조회/삭제
메시지 저장/조회
LangGraph 영구 체크포인트 적용
Chat Application 구현
```

이 단계가 완료되면 ChatGPT처럼 세션별 대화를 생성하고, 이전 대화를 다시 열고, 이어서 대화하며, 세션을 삭제할 수 있다.

이후 Admin Application과 RAG 기능을 순차적으로 추가한다.
