# 문서 검색 및 삭제 SQL

## 1. 문서 벡터 검색 SQL

아래 쿼리는 사용자 질문의 임베딩 벡터와 `document_chunks.embedding`을 비교하여 가장 유사한 문서 청크를 조회한다.

검색 대상은 다음 조건을 만족해야 한다.

* 임베딩이 존재해야 한다.
* 문서가 활성 상태여야 한다.
* 삭제되지 않은 문서여야 한다.
* 문서 처리 상태가 `READY`여야 한다.
* 전역 문서이거나 현재 세션에 연결된 문서여야 한다.

```sql
SELECT
    dc.id AS chunk_id,
    dc.document_id,
    d.file_name,
    d.category,
    dc.chunk_index,
    dc.page_number,
    dc.content,
    dc.metadata,
    1 - (
        dc.embedding <=> CAST(:query_embedding AS vector)
    ) AS similarity
FROM document_chunks dc
JOIN documents d
    ON d.id = dc.document_id
WHERE dc.embedding IS NOT NULL
  AND d.is_active = TRUE
  AND d.deleted_at IS NULL
  AND d.status = 'READY'
  AND (
        d.scope = 'GLOBAL'
        OR (
            d.scope = 'SESSION'
            AND d.session_id = :session_id
        )
  )
ORDER BY
    dc.embedding <=> CAST(:query_embedding AS vector)
LIMIT :top_k;
```

### 입력 파라미터

```text
query_embedding
  사용자 질문을 임베딩한 벡터

session_id
  현재 Chat 세션 ID

top_k
  가져올 문서 청크 개수
```

### 주요 반환값

```text
chunk_id
document_id
file_name
category
chunk_index
page_number
content
metadata
similarity
```

`<=>` 연산자는 cosine distance를 계산한다.

유사도는 다음 방식으로 변환한다.

```sql
1 - cosine_distance
```

따라서 `similarity` 값이 클수록 사용자 질문과 관련성이 높다.

---

## 2. 전역 문서만 검색하는 SQL

모든 Chat 세션에서 공통으로 사용할 전역 문서만 검색하려면 다음 쿼리를 사용한다.

```sql
SELECT
    dc.id AS chunk_id,
    dc.document_id,
    d.file_name,
    d.category,
    dc.chunk_index,
    dc.page_number,
    dc.content,
    dc.metadata,
    1 - (
        dc.embedding <=> CAST(:query_embedding AS vector)
    ) AS similarity
FROM document_chunks dc
JOIN documents d
    ON d.id = dc.document_id
WHERE dc.embedding IS NOT NULL
  AND d.is_active = TRUE
  AND d.deleted_at IS NULL
  AND d.status = 'READY'
  AND d.scope = 'GLOBAL'
ORDER BY
    dc.embedding <=> CAST(:query_embedding AS vector)
LIMIT :top_k;
```

---

## 3. 특정 카테고리 벡터 검색 SQL

특정 업무 카테고리만 대상으로 검색하려면 다음 조건을 추가한다.

```sql
SELECT
    dc.id AS chunk_id,
    dc.document_id,
    d.file_name,
    d.category,
    dc.chunk_index,
    dc.page_number,
    dc.content,
    dc.metadata,
    1 - (
        dc.embedding <=> CAST(:query_embedding AS vector)
    ) AS similarity
FROM document_chunks dc
JOIN documents d
    ON d.id = dc.document_id
WHERE dc.embedding IS NOT NULL
  AND d.is_active = TRUE
  AND d.deleted_at IS NULL
  AND d.status = 'READY'
  AND d.category = :category
  AND (
        d.scope = 'GLOBAL'
        OR (
            d.scope = 'SESSION'
            AND d.session_id = :session_id
        )
  )
ORDER BY
    dc.embedding <=> CAST(:query_embedding AS vector)
LIMIT :top_k;
```

### 추가 입력 파라미터

```text
category
  검색할 문서 카테고리

예:
  SAC
  Job Monitor
  Architecture
  AI
```

---

# 4. 키워드 검색 SQL

`document_chunks.search_vector`를 사용하여 PostgreSQL 전문 검색을 수행한다.

```sql
SELECT
    dc.id AS chunk_id,
    dc.document_id,
    d.file_name,
    d.category,
    dc.chunk_index,
    dc.page_number,
    dc.content,
    dc.metadata,
    ts_rank_cd(
        dc.search_vector,
        plainto_tsquery('simple', :query_text)
    ) AS keyword_score
FROM document_chunks dc
JOIN documents d
    ON d.id = dc.document_id
WHERE dc.search_vector
      @@ plainto_tsquery('simple', :query_text)
  AND d.is_active = TRUE
  AND d.deleted_at IS NULL
  AND d.status = 'READY'
  AND (
        d.scope = 'GLOBAL'
        OR (
            d.scope = 'SESSION'
            AND d.session_id = :session_id
        )
  )
ORDER BY keyword_score DESC
LIMIT :top_k;
```

### 입력 파라미터

```text
query_text
  사용자의 원본 질문 또는 키워드 검색용 텍스트

session_id
  현재 Chat 세션 ID

top_k
  가져올 결과 개수
```

키워드 검색은 다음과 같은 질문에 유용하다.

```text
정확한 제품명
기능명
클래스명
에러 코드
Jira 번호
업무 용어
약어
```

---

# 5. 하이브리드 검색 SQL

하이브리드 검색은 다음 두 결과를 결합한다.

```text
벡터 의미 검색
+
PostgreSQL 키워드 검색
```

아래 쿼리는 Reciprocal Rank Fusion 방식으로 두 검색 순위를 결합한다.

```sql
WITH vector_results AS (
    SELECT
        dc.id,
        ROW_NUMBER() OVER (
            ORDER BY
                dc.embedding
                <=> CAST(:query_embedding AS vector)
        ) AS vector_rank
    FROM document_chunks dc
    JOIN documents d
        ON d.id = dc.document_id
    WHERE dc.embedding IS NOT NULL
      AND d.is_active = TRUE
      AND d.deleted_at IS NULL
      AND d.status = 'READY'
      AND (
            d.scope = 'GLOBAL'
            OR (
                d.scope = 'SESSION'
                AND d.session_id = :session_id
            )
      )
    ORDER BY
        dc.embedding <=> CAST(:query_embedding AS vector)
    LIMIT :candidate_limit
),
keyword_results AS (
    SELECT
        dc.id,
        ROW_NUMBER() OVER (
            ORDER BY
                ts_rank_cd(
                    dc.search_vector,
                    plainto_tsquery('simple', :query_text)
                ) DESC
        ) AS keyword_rank
    FROM document_chunks dc
    JOIN documents d
        ON d.id = dc.document_id
    WHERE dc.search_vector
          @@ plainto_tsquery('simple', :query_text)
      AND d.is_active = TRUE
      AND d.deleted_at IS NULL
      AND d.status = 'READY'
      AND (
            d.scope = 'GLOBAL'
            OR (
                d.scope = 'SESSION'
                AND d.session_id = :session_id
            )
      )
    ORDER BY
        ts_rank_cd(
            dc.search_vector,
            plainto_tsquery('simple', :query_text)
        ) DESC
    LIMIT :candidate_limit
)
SELECT
    dc.id AS chunk_id,
    dc.document_id,
    d.file_name,
    d.category,
    dc.page_number,
    dc.chunk_index,
    dc.content,
    dc.metadata,

    COALESCE(
        1.0 / (:rrf_k + vr.vector_rank),
        0
    )
    +
    COALESCE(
        1.0 / (:rrf_k + kr.keyword_rank),
        0
    ) AS combined_score

FROM document_chunks dc
JOIN documents d
    ON d.id = dc.document_id

LEFT JOIN vector_results vr
    ON vr.id = dc.id

LEFT JOIN keyword_results kr
    ON kr.id = dc.id

WHERE vr.id IS NOT NULL
   OR kr.id IS NOT NULL

ORDER BY combined_score DESC
LIMIT :top_k;
```

### 입력 파라미터

```text
query_embedding
  사용자 질문의 임베딩 벡터

query_text
  사용자의 원본 질문

session_id
  현재 Chat 세션 ID

candidate_limit
  벡터 검색과 키워드 검색에서 각각 가져올 후보 수

rrf_k
  Reciprocal Rank Fusion 보정값

top_k
  최종 반환 결과 개수
```

권장 초기값은 다음과 같다.

```text
candidate_limit = 20
rrf_k = 60
top_k = 5
```

---

# 6. 특정 문서 내부 벡터 검색 SQL

Admin 화면이나 특정 문서 기반 질의에서 하나의 문서 안에서만 검색하려면 다음 쿼리를 사용한다.

```sql
SELECT
    dc.id AS chunk_id,
    dc.document_id,
    d.file_name,
    dc.chunk_index,
    dc.page_number,
    dc.content,
    dc.metadata,
    1 - (
        dc.embedding <=> CAST(:query_embedding AS vector)
    ) AS similarity
FROM document_chunks dc
JOIN documents d
    ON d.id = dc.document_id
WHERE dc.document_id = :document_id
  AND dc.embedding IS NOT NULL
  AND d.is_active = TRUE
  AND d.deleted_at IS NULL
  AND d.status = 'READY'
ORDER BY
    dc.embedding <=> CAST(:query_embedding AS vector)
LIMIT :top_k;
```

---

# 7. 문서 목록 조회 SQL

Admin Application에서 문서 목록을 조회할 때 사용한다.

```sql
SELECT
    d.id,
    d.file_name,
    d.original_file_name,
    d.mime_type,
    d.file_extension,
    d.file_size_bytes,
    d.category,
    d.scope,
    d.status,
    d.is_active,
    d.chunk_count,
    d.page_count,
    d.chunk_size,
    d.chunk_overlap,
    d.embedding_model,
    d.embedding_dimension,
    d.error_message,
    d.created_at,
    d.updated_at,
    d.processed_at
FROM documents d
WHERE d.deleted_at IS NULL
ORDER BY d.created_at DESC;
```

---

# 8. 활성 문서 목록 조회 SQL

RAG 검색에 실제로 사용 가능한 문서만 조회한다.

```sql
SELECT
    d.id,
    d.file_name,
    d.category,
    d.scope,
    d.status,
    d.chunk_count,
    d.page_count,
    d.embedding_model,
    d.created_at,
    d.processed_at
FROM documents d
WHERE d.is_active = TRUE
  AND d.deleted_at IS NULL
  AND d.status = 'READY'
ORDER BY d.updated_at DESC;
```

---

# 9. 문서 상세 조회 SQL

문서 하나의 상세 정보와 청크 수를 조회한다.

```sql
SELECT
    d.id,
    d.uploaded_by,
    d.file_name,
    d.original_file_name,
    d.original_path,
    d.extracted_text_path,
    d.mime_type,
    d.file_extension,
    d.file_size_bytes,
    d.file_hash_sha256,
    d.category,
    d.scope,
    d.session_id,
    d.status,
    d.is_active,
    d.chunk_count,
    d.page_count,
    d.chunk_size,
    d.chunk_overlap,
    d.embedding_model,
    d.embedding_dimension,
    d.error_message,
    d.metadata,
    d.created_at,
    d.updated_at,
    d.processed_at,
    d.deleted_at,
    COUNT(dc.id) AS actual_chunk_count
FROM documents d
LEFT JOIN document_chunks dc
    ON dc.document_id = d.id
WHERE d.id = :document_id
GROUP BY d.id;
```

---

# 10. 문서 청크 조회 SQL

Admin 화면에서 청크를 페이지 단위로 조회한다.

```sql
SELECT
    dc.id,
    dc.document_id,
    dc.chunk_index,
    dc.page_number,
    dc.page_chunk_index,
    dc.start_offset,
    dc.end_offset,
    dc.content,
    dc.token_count,
    dc.metadata,
    dc.created_at
FROM document_chunks dc
WHERE dc.document_id = :document_id
ORDER BY
    dc.chunk_index ASC
LIMIT :page_size
OFFSET :offset;
```

---

# 11. 문서 업로드 레코드 생성 SQL

파일이 업로드되면 먼저 `documents` 테이블에 레코드를 생성한다.

```sql
INSERT INTO documents (
    uploaded_by,
    file_name,
    original_file_name,
    original_path,
    mime_type,
    file_extension,
    file_size_bytes,
    file_hash_sha256,
    category,
    scope,
    session_id,
    status,
    chunk_size,
    chunk_overlap,
    metadata
)
VALUES (
    :uploaded_by,
    :file_name,
    :original_file_name,
    :original_path,
    :mime_type,
    :file_extension,
    :file_size_bytes,
    :file_hash_sha256,
    :category,
    :scope,
    :session_id,
    'UPLOADED',
    :chunk_size,
    :chunk_overlap,
    CAST(:metadata AS JSONB)
)
RETURNING id;
```

---

# 12. 문서 상태 변경 SQL

문서 처리 단계가 바뀔 때 사용한다.

```sql
UPDATE documents
SET
    status = :status,
    error_message = :error_message,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :document_id;
```

예시 상태:

```text
UPLOADED
EXTRACTING
EXTRACTED
CHUNKING
CHUNKED
EMBEDDING
READY
FAILED
INACTIVE
DELETED
```

---

# 13. 문서 처리 완료 SQL

청크와 임베딩 저장이 모두 완료되면 문서를 `READY` 상태로 변경한다.

```sql
UPDATE documents
SET
    status = 'READY',
    is_active = TRUE,
    chunk_count = :chunk_count,
    page_count = :page_count,
    embedding_model = :embedding_model,
    embedding_dimension = :embedding_dimension,
    processed_at = CURRENT_TIMESTAMP,
    error_message = NULL,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :document_id;
```

---

# 14. 문서 처리 실패 SQL

```sql
UPDATE documents
SET
    status = 'FAILED',
    is_active = FALSE,
    error_message = :error_message,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :document_id;
```

---

# 15. 문서 청크 저장 SQL

각 청크와 임베딩을 저장한다.

```sql
INSERT INTO document_chunks (
    document_id,
    chunk_index,
    page_number,
    page_chunk_index,
    start_offset,
    end_offset,
    content,
    token_count,
    embedding,
    metadata
)
VALUES (
    :document_id,
    :chunk_index,
    :page_number,
    :page_chunk_index,
    :start_offset,
    :end_offset,
    :content,
    :token_count,
    CAST(:embedding AS vector),
    CAST(:metadata AS JSONB)
)
RETURNING id;
```

---

# 16. 문서 청크 일괄 재생성을 위한 삭제 SQL

재벡터화 또는 재청킹 전에 기존 청크를 삭제한다.

```sql
DELETE FROM document_chunks
WHERE document_id = :document_id;
```

그다음 새 청크를 다시 저장한다.

---

# 17. 문서 비활성화 SQL

문서를 삭제하지 않고 RAG 검색 대상에서만 제외한다.

```sql
UPDATE documents
SET
    is_active = FALSE,
    status = 'INACTIVE',
    updated_at = CURRENT_TIMESTAMP
WHERE id = :document_id
  AND deleted_at IS NULL;
```

비활성화된 문서는 다음 조건 때문에 검색에서 제외된다.

```sql
d.is_active = TRUE
```

---

# 18. 문서 재활성화 SQL

```sql
UPDATE documents
SET
    is_active = TRUE,
    status = 'READY',
    updated_at = CURRENT_TIMESTAMP
WHERE id = :document_id
  AND deleted_at IS NULL
  AND chunk_count > 0;
```

---

# 19. 문서 Soft Delete SQL

문서를 즉시 물리 삭제하지 않고 삭제 상태로 변경한다.

```sql
UPDATE documents
SET
    is_active = FALSE,
    status = 'DELETED',
    deleted_at = CURRENT_TIMESTAMP,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :document_id
  AND deleted_at IS NULL;
```

Soft Delete된 문서는 DB에는 남아 있지만 RAG 검색에서는 제외된다.

---

# 20. 문서 물리 삭제 SQL

문서와 연결된 DB 데이터를 완전히 삭제한다.

```sql
BEGIN;

DELETE FROM documents
WHERE id = :document_id;

COMMIT;
```

다음 테이블은 외래키의 `ON DELETE CASCADE`에 의해 자동 삭제된다.

```text
document_chunks
document_tags
ingestion_jobs
```

원본 파일과 추출 텍스트 파일은 Python 애플리케이션에서 별도로 삭제해야 한다.

삭제 대상 예시:

```text
data/uploads/{document_id}/original.pdf
data/extracted/{document_id}/extracted.txt
```

---

# 21. 문서 태그 생성 SQL

```sql
INSERT INTO tags (
    name,
    description
)
VALUES (
    :name,
    :description
)
ON CONFLICT (name)
DO UPDATE SET
    description = COALESCE(
        EXCLUDED.description,
        tags.description
    )
RETURNING id;
```

---

# 22. 문서에 태그 연결 SQL

```sql
INSERT INTO document_tags (
    document_id,
    tag_id
)
VALUES (
    :document_id,
    :tag_id
)
ON CONFLICT (
    document_id,
    tag_id
)
DO NOTHING;
```

---

# 23. 문서 태그 제거 SQL

```sql
DELETE FROM document_tags
WHERE document_id = :document_id
  AND tag_id = :tag_id;
```

---

# 24. 문서 태그 조회 SQL

```sql
SELECT
    t.id,
    t.name,
    t.description
FROM document_tags dt
JOIN tags t
    ON t.id = dt.tag_id
WHERE dt.document_id = :document_id
ORDER BY t.name ASC;
```

---

# 25. Ingestion Job 생성 SQL

문서 처리 작업을 생성한다.

```sql
INSERT INTO ingestion_jobs (
    document_id,
    job_type,
    status,
    progress_percent,
    current_step
)
VALUES (
    :document_id,
    :job_type,
    'PENDING',
    0,
    :current_step
)
RETURNING id;
```

작업 유형 예시:

```text
EXTRACT
OCR
CHUNK
EMBED
REINDEX
DELETE
```

---

# 26. Ingestion Job 시작 SQL

```sql
UPDATE ingestion_jobs
SET
    status = 'RUNNING',
    started_at = CURRENT_TIMESTAMP,
    attempt_count = attempt_count + 1,
    current_step = :current_step,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :job_id;
```

---

# 27. Ingestion Job 진행률 변경 SQL

```sql
UPDATE ingestion_jobs
SET
    progress_percent = :progress_percent,
    current_step = :current_step,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :job_id;
```

---

# 28. Ingestion Job 완료 SQL

```sql
UPDATE ingestion_jobs
SET
    status = 'COMPLETED',
    progress_percent = 100,
    completed_at = CURRENT_TIMESTAMP,
    error_message = NULL,
    error_details = NULL,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :job_id;
```

---

# 29. Ingestion Job 실패 SQL

```sql
UPDATE ingestion_jobs
SET
    status = 'FAILED',
    completed_at = CURRENT_TIMESTAMP,
    error_message = :error_message,
    error_details = CAST(:error_details AS JSONB),
    updated_at = CURRENT_TIMESTAMP
WHERE id = :job_id;
```

---

# 30. 세션 생성 SQL

```sql
INSERT INTO chat_sessions (
    user_id,
    title,
    default_mode,
    last_message_at
)
VALUES (
    :user_id,
    :title,
    :default_mode,
    CURRENT_TIMESTAMP
)
RETURNING id;
```

초기 제목 예시:

```text
새 대화
```

---

# 31. 세션 목록 조회 SQL

```sql
SELECT
    s.id,
    s.title,
    s.default_mode,
    s.summary,
    s.is_pinned,
    s.is_archived,
    s.last_message_at,
    s.created_at,
    s.updated_at,
    COUNT(m.id) AS message_count
FROM chat_sessions s
LEFT JOIN chat_messages m
    ON m.session_id = s.id
WHERE s.user_id = :user_id
  AND s.deleted_at IS NULL
  AND s.is_archived = FALSE
GROUP BY s.id
ORDER BY
    s.is_pinned DESC,
    COALESCE(
        s.last_message_at,
        s.created_at
    ) DESC;
```

---

# 32. 세션 제목 변경 SQL

```sql
UPDATE chat_sessions
SET
    title = :title,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :session_id
  AND deleted_at IS NULL;
```

---

# 33. 세션 Soft Delete SQL

```sql
UPDATE chat_sessions
SET
    deleted_at = CURRENT_TIMESTAMP,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :session_id
  AND deleted_at IS NULL;
```

Soft Delete 방식에서는 메시지도 실제로 삭제되지 않는다.

---

# 34. 세션 물리 삭제 SQL

```sql
BEGIN;

DELETE FROM chat_sessions
WHERE id = :session_id;

COMMIT;
```

`chat_messages`는 `ON DELETE CASCADE`로 함께 삭제된다.

LangGraph PostgreSQL checkpointer에 저장된 동일한 `thread_id` 데이터는 애플리케이션 코드에서 별도로 삭제해야 한다.

```text
chat_sessions.id
=
LangGraph thread_id
```

---

# 35. 메시지 저장 SQL

```sql
INSERT INTO chat_messages (
    session_id,
    role,
    content,
    mode,
    status,
    model_name,
    prompt_tokens,
    completion_tokens,
    metadata
)
VALUES (
    :session_id,
    :role,
    :content,
    :mode,
    :status,
    :model_name,
    :prompt_tokens,
    :completion_tokens,
    CAST(:metadata AS JSONB)
)
RETURNING id;
```

---

# 36. 세션 메시지 조회 SQL

```sql
SELECT
    id,
    session_id,
    role,
    content,
    mode,
    status,
    model_name,
    prompt_tokens,
    completion_tokens,
    metadata,
    created_at
FROM chat_messages
WHERE session_id = :session_id
ORDER BY created_at ASC;
```

---

# 37. 최근 메시지 조회 SQL

LLM에 전달할 최근 대화만 조회한다.

```sql
SELECT
    id,
    role,
    content,
    mode,
    metadata,
    created_at
FROM chat_messages
WHERE session_id = :session_id
  AND status = 'COMPLETE'
ORDER BY created_at DESC
LIMIT :message_limit;
```

애플리케이션에서는 결과를 다시 시간 오름차순으로 정렬해 LLM에 전달한다.

---

# 38. 세션 마지막 메시지 시간 갱신 SQL

```sql
UPDATE chat_sessions
SET
    last_message_at = CURRENT_TIMESTAMP,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :session_id;
```

---

# 39. 세션 요약 저장 SQL

긴 대화를 요약하여 저장한다.

```sql
UPDATE chat_sessions
SET
    summary = :summary,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :session_id;
```

---

# 40. 장기 기억 저장 SQL

```sql
INSERT INTO personal_memories (
    user_id,
    memory_type,
    title,
    content,
    embedding,
    importance,
    source_session_id,
    source_message_id,
    metadata
)
VALUES (
    :user_id,
    :memory_type,
    :title,
    :content,
    CAST(:embedding AS vector),
    :importance,
    :source_session_id,
    :source_message_id,
    CAST(:metadata AS JSONB)
)
RETURNING id;
```

---

# 41. 장기 기억 벡터 검색 SQL

```sql
SELECT
    pm.id,
    pm.memory_type,
    pm.title,
    pm.content,
    pm.importance,
    pm.metadata,
    1 - (
        pm.embedding <=> CAST(:query_embedding AS vector)
    ) AS similarity
FROM personal_memories pm
WHERE pm.user_id = :user_id
  AND pm.is_active = TRUE
  AND pm.deleted_at IS NULL
  AND pm.embedding IS NOT NULL
ORDER BY
    pm.embedding <=> CAST(:query_embedding AS vector)
LIMIT :top_k;
```

---

# 42. 장기 기억 비활성화 SQL

```sql
UPDATE personal_memories
SET
    is_active = FALSE,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :memory_id;
```

---

# 43. 장기 기억 Soft Delete SQL

```sql
UPDATE personal_memories
SET
    is_active = FALSE,
    deleted_at = CURRENT_TIMESTAMP,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :memory_id;
```

---

# 44. LangGraph Checkpointer 초기화

LangGraph 내부 체크포인트 테이블은 직접 DDL을 작성하지 않는다.

설치된 LangGraph 버전에 맞게 `PostgresSaver.setup()`을 실행하여 생성한다.

```python
from langgraph.checkpoint.postgres import PostgresSaver


DB_URI = (
    "postgresql://assistant:assistant_password"
    "@localhost:5432/local_assistant"
)


with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
```

필요한 패키지:

```bash
pip install langgraph-checkpoint-postgres "psycopg[binary,pool]"
```

---

# 45. SQL 파일 적용 방법

파일 예시:

```text
docker/postgres/init/02-schema.sql
```

기존 DB에 직접 적용:

```bash
docker compose exec -T postgres \
  psql \
  -U assistant \
  -d local_assistant \
  < docker/postgres/init/02-schema.sql
```

테이블 확인:

```bash
docker compose exec postgres \
  psql \
  -U assistant \
  -d local_assistant
```

PostgreSQL 접속 후:

```sql
\dt
```

pgvector 확인:

```sql
SELECT
    extname,
    extversion
FROM pg_extension
WHERE extname = 'vector';
```

주요 테이블 확인:

```text
app_users
chat_sessions
chat_messages
documents
document_chunks
tags
document_tags
ingestion_jobs
personal_memories
```
