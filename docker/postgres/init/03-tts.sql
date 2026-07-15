-- ============================================================
-- TTS / 사용자 설정 테이블
-- ============================================================

BEGIN;

-- ──────────────────────────────────────────────
-- 사용자 TTS 설정
-- ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id UUID PRIMARY KEY,

    tts_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    tts_autoplay BOOLEAN NOT NULL DEFAULT FALSE,

    -- BROWSER: Web Speech API
    -- MACOS: macOS say 명령
    -- PIPER: Piper TTS
    -- KOKORO: Kokoro TTS
    tts_engine VARCHAR(30) NOT NULL DEFAULT 'BROWSER',

    tts_voice VARCHAR(200),

    tts_language VARCHAR(20) NOT NULL DEFAULT 'ko-KR',

    tts_rate NUMERIC(4, 2) NOT NULL DEFAULT 1.0,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_user_preferences_user
        FOREIGN KEY (user_id)
        REFERENCES app_users(id)
        ON DELETE CASCADE,

    CONSTRAINT ck_user_preferences_tts_engine
        CHECK (
            tts_engine IN (
                'BROWSER',
                'MACOS',
                'PIPER',
                'KOKORO'
            )
        ),

    CONSTRAINT ck_user_preferences_tts_rate
        CHECK (tts_rate BETWEEN 0.5 AND 2.0)
);

CREATE TRIGGER trg_user_preferences_updated_at
BEFORE UPDATE ON user_preferences
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();


-- ──────────────────────────────────────────────
-- 메시지별 오디오 파일 캐시
-- ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS message_audio (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    message_id UUID NOT NULL,

    engine VARCHAR(30) NOT NULL,

    voice VARCHAR(200),

    language VARCHAR(20),

    audio_path TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_message_audio_message
        FOREIGN KEY (message_id)
        REFERENCES chat_messages(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_message_audio_message
    ON message_audio (message_id);

COMMIT;
