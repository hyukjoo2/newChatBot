-- ============================================================
-- Local AI Assistant
-- PostgreSQL + pgvector Initial Schema
-- ============================================================

BEGIN;


-- ============================================================
-- 1. Extensions
-- ============================================================

-- Vector 타입과 벡터 검색 기능
CREATE EXTENSION IF NOT EXISTS vector;

-- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pgcrypto;


-- ============================================================
-- 2. Common updated_at trigger
-- ============================================================

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;


-- ============================================================
-- 3. Application users
-- ============================================================
-- 현재는 개인용이지만, Chat/Admin 권한 확장을 고려한 테이블이다.

CREATE TABLE IF NOT EXISTS app_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    username VARCHAR(100) NOT NULL,
    display_name VARCHAR(200),
    email VARCHAR(320),

    -- USER: Chat Application 사용
    -- ADMIN: Chat + Admin Application 사용
    role VARCHAR(20) NOT NULL DEFAULT 'USER',

    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_app_users_username
        UNIQUE (username),

    CONSTRAINT uq_app_users_email
        UNIQUE (email),

    CONSTRAINT ck_app_users_role
        CHECK (role IN ('USER', 'ADMIN'))
);

CREATE TRIGGER trg_app_users_updated_at
BEFORE UPDATE ON app_users
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();


-- 초기 관리자 사용자
-- 인증 기능을 붙이기 전까지 사용할 기본 사용자다.
INSERT INTO app_users (
    username,
    display_name,
    role
)
VALUES (
    'admin',
    'Administrator',
    'ADMIN'
)
ON CONFLICT (username) DO NOTHING;


-- ============================================================
-- 4. Chat sessions
-- ============================================================

CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    user_id UUID NOT NULL,

    title VARCHAR(500) NOT NULL DEFAULT '새 대화',

    -- CHAT: 일반 대화
    -- RAG: 지식 검색
    -- AUTO: 향후 자동 분류 모드
    default_mode VARCHAR(20) NOT NULL DEFAULT 'CHAT',

    -- 긴 대화에서 이전 내용을 요약해 저장
    summary TEXT,

    -- 마지막 메시지 시각
    last_message_at TIMESTAMPTZ,

    is_pinned BOOLEAN NOT NULL DEFAULT FALSE,
    is_archived BOOLEAN NOT NULL DEFAULT FALSE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ,

    CONSTRAINT fk_chat_sessions_user
        FOREIGN KEY (user_id)
        REFERENCES app_users(id)
        ON DELETE CASCADE,

    CONSTRAINT ck_chat_sessions_default_mode
        CHECK (default_mode IN ('CHAT', 'RAG', 'AUTO'))
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
    ON chat_sessions (
        user_id,
        updated_at DESC
    )
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_last_message
    ON chat_sessions (
        user_id,
        last_message_at DESC
    )
    WHERE deleted_at IS NULL;

CREATE TRIGGER trg_chat_sessions_updated_at
BEFORE UPDATE ON chat_sessions
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();


-- ============================================================
-- 5. Chat messages
-- ============================================================

CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    session_id UUID NOT NULL,

    -- USER, ASSISTANT, SYSTEM, TOOL
    role VARCHAR(20) NOT NULL,

    content TEXT NOT NULL,

    -- 실제 해당 메시지 생성에 사용된 모드
    mode VARCHAR(20),

    -- COMPLETE, STREAMING, FAILED, CANCELLED
    status VARCHAR(20) NOT NULL DEFAULT 'COMPLETE',

    -- 모델명
    model_name VARCHAR(200),

    -- 입력/출력 토큰 수
    prompt_tokens INTEGER,
    completion_tokens INTEGER,

    -- RAG 출처, 오류, 모델 옵션 등
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_chat_messages_session
        FOREIGN KEY (session_id)
        REFERENCES chat_sessions(id)
        ON DELETE CASCADE,

    CONSTRAINT ck_chat_messages_role
        CHECK (
            role IN (
                'USER',
                'ASSISTANT',
                'SYSTEM',
                'TOOL'
            )
        ),

    CONSTRAINT ck_chat_messages_mode
        CHECK (
            mode IS NULL
            OR mode IN ('CHAT', 'RAG', 'AUTO')
        ),

    CONSTRAINT ck_chat_messages_status
        CHECK (
            status IN (
                'COMPLETE',
                'STREAMING',
                'FAILED',
                'CANCELLED'
            )
        ),

    CONSTRAINT ck_chat_messages_prompt_tokens
        CHECK (
            prompt_tokens IS NULL
            OR prompt_tokens >= 0
        ),

    CONSTRAINT ck_chat_messages_completion_tokens
        CHECK (
            completion_tokens IS NULL
            OR completion_tokens >= 0
        )
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
    ON chat_messages (
        session_id,
        created_at ASC
    );

CREATE INDEX IF NOT EXISTS idx_chat_messages_metadata_gin
    ON chat_messages
    USING GIN (metadata);


-- ============================================================
-- 6. Documents
-- ============================================================

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    uploaded_by UUID NOT NULL,

    file_name VARCHAR(1000) NOT NULL,
    original_file_name VARCHAR(1000) NOT NULL,

    original_path TEXT NOT NULL,
    extracted_text_path TEXT,

    mime_type VARCHAR(255),
    file_extension VARCHAR(50),

    file_size_bytes BIGINT,
    file_hash_sha256 VARCHAR(64),

    category VARCHAR(200),

    -- GLOBAL: 모든 세션에서 검색 가능
    -- SESSION: 특정 세션에서만 검색 가능
    scope VARCHAR(20) NOT NULL DEFAULT 'GLOBAL',

    session_id UUID,

    -- 문서 파이프라인 상태
    status VARCHAR(30) NOT NULL DEFAULT 'UPLOADED',

    -- 검색 대상 여부
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    chunk_count INTEGER NOT NULL DEFAULT 0,
    page_count INTEGER,

    chunk_size INTEGER,
    chunk_overlap INTEGER,

    embedding_model VARCHAR(200),
    embedding_dimension INTEGER,

    error_message TEXT,

    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,

    CONSTRAINT fk_documents_uploaded_by
        FOREIGN KEY (uploaded_by)
        REFERENCES app_users(id)
        ON DELETE RESTRICT,

    CONSTRAINT fk_documents_session
        FOREIGN KEY (session_id)
        REFERENCES chat_sessions(id)
        ON DELETE CASCADE,

    CONSTRAINT uq_documents_file_hash
        UNIQUE (file_hash_sha256),

    CONSTRAINT ck_documents_scope
        CHECK (
            scope IN ('GLOBAL', 'SESSION')
        ),

    CONSTRAINT ck_documents_scope_session
        CHECK (
            (scope = 'GLOBAL' AND session_id IS NULL)
            OR
            (scope = 'SESSION' AND session_id IS NOT NULL)
        ),

    CONSTRAINT ck_documents_status
        CHECK (
            status IN (
                'UPLOADED',
                'EXTRACTING',
                'EXTRACTED',
                'CHUNKING',
                'CHUNKED',
                'EMBEDDING',
                'READY',
                'FAILED',
                'INACTIVE',
                'DELETED'
            )
        ),

    CONSTRAINT ck_documents_file_size
        CHECK (
            file_size_bytes IS NULL
            OR file_size_bytes >= 0
        ),

    CONSTRAINT ck_documents_chunk_count
        CHECK (chunk_count >= 0),

    CONSTRAINT ck_documents_page_count
        CHECK (
            page_count IS NULL
            OR page_count >= 0
        ),

    CONSTRAINT ck_documents_chunk_size
        CHECK (
            chunk_size IS NULL
            OR chunk_size > 0
        ),

    CONSTRAINT ck_documents_chunk_overlap
        CHECK (
            chunk_overlap IS NULL
            OR chunk_overlap >= 0
        )
);

CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents (status);

CREATE INDEX IF NOT EXISTS idx_documents_active_status
    ON documents (
        is_active,
        status
    )
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_documents_category
    ON documents (category)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_documents_scope_session
    ON documents (
        scope,
        session_id
    )
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_documents_created_at
    ON documents (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin
    ON documents
    USING GIN (metadata);

CREATE TRIGGER trg_documents_updated_at
BEFORE UPDATE ON documents
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();


-- ============================================================
-- 7. Document chunks and embeddings
-- ============================================================
-- nomic-embed-text 기준 VECTOR(768)
-- 다른 모델 사용 시 차원을 변경해야 한다.

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    document_id UUID NOT NULL,

    chunk_index INTEGER NOT NULL,

    page_number INTEGER,

    -- 페이지 내부 청크 순서
    page_chunk_index INTEGER,

    -- 원문 내 문자 위치
    start_offset INTEGER,
    end_offset INTEGER,

    content TEXT NOT NULL,

    token_count INTEGER,

    -- nomic-embed-text의 임베딩 차원
    embedding VECTOR(768),

    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,

    -- PostgreSQL 키워드 검색용
    search_vector TSVECTOR
        GENERATED ALWAYS AS (
            to_tsvector(
                'simple',
                COALESCE(content, '')
            )
        ) STORED,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_document_chunks_document
        FOREIGN KEY (document_id)
        REFERENCES documents(id)
        ON DELETE CASCADE,

    CONSTRAINT uq_document_chunks_index
        UNIQUE (
            document_id,
            chunk_index
        ),

    CONSTRAINT ck_document_chunks_chunk_index
        CHECK (chunk_index >= 0),

    CONSTRAINT ck_document_chunks_page_number
        CHECK (
            page_number IS NULL
            OR page_number > 0
        ),

    CONSTRAINT ck_document_chunks_page_chunk_index
        CHECK (
            page_chunk_index IS NULL
            OR page_chunk_index >= 0
        ),

    CONSTRAINT ck_document_chunks_offsets
        CHECK (
            (
                start_offset IS NULL
                AND end_offset IS NULL
            )
            OR
            (
                start_offset IS NOT NULL
                AND end_offset IS NOT NULL
                AND start_offset >= 0
                AND end_offset >= start_offset
            )
        ),

    CONSTRAINT ck_document_chunks_content_not_empty
        CHECK (length(trim(content)) > 0),

    CONSTRAINT ck_document_chunks_token_count
        CHECK (
            token_count IS NULL
            OR token_count >= 0
        )
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document
    ON document_chunks (
        document_id,
        chunk_index
    );

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_page
    ON document_chunks (
        document_id,
        page_number,
        chunk_index
    );

CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata_gin
    ON document_chunks
    USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_document_chunks_search_vector
    ON document_chunks
    USING GIN (search_vector);


-- ============================================================
-- 8. Vector similarity index
-- ============================================================
-- cosine distance 검색용 HNSW 인덱스
--
-- 검색 SQL에서도 반드시 <=> 연산자를 사용해야 한다.
--
-- 데이터가 적은 초기 개발 단계에서는 없어도 동작하지만,
-- 문서 청크가 많아지면 검색 속도 향상에 도움을 준다.

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
    ON document_chunks
    USING HNSW (
        embedding vector_cosine_ops
    )
    WHERE embedding IS NOT NULL;


-- ============================================================
-- 9. Tags
-- ============================================================

CREATE TABLE IF NOT EXISTS tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    name VARCHAR(200) NOT NULL,
    description TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_tags_name
        UNIQUE (name)
);


CREATE TABLE IF NOT EXISTS document_tags (
    document_id UUID NOT NULL,
    tag_id UUID NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (
        document_id,
        tag_id
    ),

    CONSTRAINT fk_document_tags_document
        FOREIGN KEY (document_id)
        REFERENCES documents(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_document_tags_tag
        FOREIGN KEY (tag_id)
        REFERENCES tags(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_document_tags_tag
    ON document_tags (tag_id);


-- ============================================================
-- 10. Document ingestion jobs
-- ============================================================

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    document_id UUID NOT NULL,

    -- EXTRACT, OCR, CHUNK, EMBED, REINDEX, DELETE
    job_type VARCHAR(30) NOT NULL,

    -- PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    status VARCHAR(30) NOT NULL DEFAULT 'PENDING',

    progress_percent NUMERIC(5, 2) NOT NULL DEFAULT 0,

    current_step TEXT,

    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,

    error_message TEXT,
    error_details JSONB,

    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_ingestion_jobs_document
        FOREIGN KEY (document_id)
        REFERENCES documents(id)
        ON DELETE CASCADE,

    CONSTRAINT ck_ingestion_jobs_job_type
        CHECK (
            job_type IN (
                'EXTRACT',
                'OCR',
                'CHUNK',
                'EMBED',
                'REINDEX',
                'DELETE'
            )
        ),

    CONSTRAINT ck_ingestion_jobs_status
        CHECK (
            status IN (
                'PENDING',
                'RUNNING',
                'COMPLETED',
                'FAILED',
                'CANCELLED'
            )
        ),

    CONSTRAINT ck_ingestion_jobs_progress
        CHECK (
            progress_percent >= 0
            AND progress_percent <= 100
        ),

    CONSTRAINT ck_ingestion_jobs_attempt_count
        CHECK (attempt_count >= 0),

    CONSTRAINT ck_ingestion_jobs_max_attempts
        CHECK (max_attempts > 0)
);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_document_created
    ON ingestion_jobs (
        document_id,
        created_at DESC
    );

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status_created
    ON ingestion_jobs (
        status,
        created_at ASC
    );

CREATE TRIGGER trg_ingestion_jobs_updated_at
BEFORE UPDATE ON ingestion_jobs
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();


-- ============================================================
-- 11. Personal long-term memories
-- ============================================================
-- 세션을 넘어 기억해야 하는 사용자 선호, 업무 사실,
-- 프로젝트 정보 등을 저장한다.

CREATE TABLE IF NOT EXISTS personal_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    user_id UUID NOT NULL,

    -- PREFERENCE, FACT, PROJECT, INSTRUCTION, OTHER
    memory_type VARCHAR(30) NOT NULL,

    title VARCHAR(500),
    content TEXT NOT NULL,

    -- 의미 검색이 필요할 경우 사용
    embedding VECTOR(768),

    importance NUMERIC(4, 3) NOT NULL DEFAULT 0.500,

    source_session_id UUID,
    source_message_id UUID,

    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,

    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,

    CONSTRAINT fk_personal_memories_user
        FOREIGN KEY (user_id)
        REFERENCES app_users(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_personal_memories_session
        FOREIGN KEY (source_session_id)
        REFERENCES chat_sessions(id)
        ON DELETE SET NULL,

    CONSTRAINT fk_personal_memories_message
        FOREIGN KEY (source_message_id)
        REFERENCES chat_messages(id)
        ON DELETE SET NULL,

    CONSTRAINT ck_personal_memories_type
        CHECK (
            memory_type IN (
                'PREFERENCE',
                'FACT',
                'PROJECT',
                'INSTRUCTION',
                'OTHER'
            )
        ),

    CONSTRAINT ck_personal_memories_content_not_empty
        CHECK (length(trim(content)) > 0),

    CONSTRAINT ck_personal_memories_importance
        CHECK (
            importance >= 0
            AND importance <= 1
        )
);

CREATE INDEX IF NOT EXISTS idx_personal_memories_user_active
    ON personal_memories (
        user_id,
        is_active
    )
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_personal_memories_type
    ON personal_memories (
        user_id,
        memory_type
    )
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_personal_memories_metadata_gin
    ON personal_memories
    USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_personal_memories_embedding_hnsw
    ON personal_memories
    USING HNSW (
        embedding vector_cosine_ops
    )
    WHERE embedding IS NOT NULL;

CREATE TRIGGER trg_personal_memories_updated_at
BEFORE UPDATE ON personal_memories
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();


-- ============================================================
-- 12. Useful views
-- ============================================================

CREATE OR REPLACE VIEW active_documents AS
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
    d.chunk_count,
    d.page_count,
    d.chunk_size,
    d.chunk_overlap,
    d.embedding_model,
    d.embedding_dimension,
    d.metadata,
    d.created_at,
    d.updated_at,
    d.processed_at
FROM documents d
WHERE d.is_active = TRUE
  AND d.deleted_at IS NULL
  AND d.status = 'READY';


CREATE OR REPLACE VIEW chat_session_summaries AS
SELECT
    s.id,
    s.user_id,
    s.title,
    s.default_mode,
    s.summary,
    s.is_pinned,
    s.is_archived,
    s.created_at,
    s.updated_at,
    s.last_message_at,
    COUNT(m.id) AS message_count
FROM chat_sessions s
LEFT JOIN chat_messages m
    ON m.session_id = s.id
WHERE s.deleted_at IS NULL
GROUP BY
    s.id,
    s.user_id,
    s.title,
    s.default_mode,
    s.summary,
    s.is_pinned,
    s.is_archived,
    s.created_at,
    s.updated_at,
    s.last_message_at;


COMMIT;