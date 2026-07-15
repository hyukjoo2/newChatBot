# 챗봇 응답 읽어주기 기능 설계

## 1. 목표

챗봇의 응답을 다음 두 가지 방식으로 동시에 제공한다.

```text
1. 화면에 텍스트로 표시
2. 음성으로 읽어주기
```

전체 처리 흐름은 다음과 같다.

```text
사용자 질문
   ↓
LangGraph
   ↓
Gemma 3 4B
   ↓
텍스트 응답 생성
   ├─ Chat 화면에 텍스트 표시
   └─ TTS 엔진으로 전달
          ↓
       음성 생성
          ↓
       브라우저 또는 오디오 플레이어에서 재생
```

---

# 2. Chat Application UI

챗봇 응답 아래에 음성 재생 기능을 제공한다.

예시:

```text
Gemma:

오늘 회의 일정은 오후 3시입니다.

[🔊 읽기] [⏹ 중지] [🔁 다시 듣기]
```

사용자 설정으로 다음 옵션을 제공할 수 있다.

```text
음성 읽기 사용
자동 읽기
응답마다 읽기 버튼 표시
말하기 속도
음성 선택
언어 선택
```

---

# 3. 구현 방식

음성 읽기 기능은 크게 두 가지 방식으로 구현할 수 있다.

```text
1. 브라우저 Web Speech API
2. 로컬 TTS 모델
```

---

# 4. 브라우저 Web Speech API

브라우저에 내장된 음성 합성 기능을 사용한다.

## 장점

```text
별도 TTS 모델이 필요하지 않다.
구현이 빠르다.
서버 부하가 거의 없다.
오디오 파일을 생성하지 않아도 된다.
맥북의 시스템 음성을 사용할 수 있다.
```

## 단점

```text
브라우저와 운영체제에 따라 음성이 달라질 수 있다.
한국어 음질이 시스템 환경에 따라 다를 수 있다.
Streamlit에서 JavaScript 컴포넌트가 필요할 수 있다.
브라우저 자동 재생 정책의 영향을 받을 수 있다.
```

## JavaScript 예시

```javascript
const utterance = new SpeechSynthesisUtterance(answerText);

utterance.lang = "ko-KR";
utterance.rate = 1.0;
utterance.pitch = 1.0;
utterance.volume = 1.0;

window.speechSynthesis.speak(utterance);
```

음성 중지:

```javascript
window.speechSynthesis.cancel();
```

## 권장 사용 시점

초기 프로토타입에서는 브라우저 Web Speech API를 사용하는 것이 가장 단순하다.

```text
1차 버전
  브라우저 Web Speech API

2차 버전
  로컬 TTS 모델

3차 버전
  음성 입력과 음성 출력 통합
```

---

# 5. 로컬 TTS 모델

챗봇 응답 텍스트를 로컬 TTS 모델에 전달하여 WAV 또는 MP3 파일을 생성한다.

```text
Gemma 응답
   ↓
Local TTS
   ↓
WAV 또는 MP3 생성
   ↓
Streamlit 오디오 플레이어
```

## 장점

```text
완전한 로컬 실행이 가능하다.
네트워크 연결이 필요 없다.
음성 품질과 모델을 직접 선택할 수 있다.
항상 동일한 음성을 사용할 수 있다.
```

## 단점

```text
추가 모델을 설치해야 한다.
메모리와 CPU를 더 사용한다.
맥북 M1 환경에서 호환성을 확인해야 한다.
한국어 음성 품질이 모델마다 다르다.
```

## 후보 TTS 엔진

```text
Piper
Kokoro
Coqui TTS
macOS say 명령
```

초기 로컬 테스트에는 macOS의 `say` 명령도 사용할 수 있다.

```bash
say -v Yuna "안녕하세요. 오늘 일정은 오후 세 시입니다."
```

오디오 파일로 저장:

```bash
say \
  -v Yuna \
  -o output.aiff \
  "안녕하세요. 오늘 일정은 오후 세 시입니다."
```

WAV로 변환하려면 `ffmpeg`를 사용할 수 있다.

```bash
ffmpeg -i output.aiff output.wav
```

---

# 6. Streamlit 오디오 재생

TTS 엔진이 WAV 파일을 생성했다고 가정한다.

```python
import streamlit as st


def render_audio(
    audio_path: str,
    autoplay: bool = False,
) -> None:
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    st.audio(
        audio_bytes,
        format="audio/wav",
        autoplay=autoplay,
    )
```

챗봇 응답 아래에 읽기 버튼을 추가할 수 있다.

```python
with st.chat_message("assistant"):
    st.markdown(answer)

    if st.button(
        "🔊 읽기",
        key=f"tts-{message_id}",
    ):
        audio_path = tts_service.synthesize(
            text=answer,
            language="ko",
        )

        render_audio(
            audio_path=audio_path,
            autoplay=True,
        )
```

---

# 7. TTS 서비스 인터페이스

백엔드가 특정 TTS 엔진에 종속되지 않도록 인터페이스를 분리한다.

```python
from pathlib import Path
from typing import Protocol


class TextToSpeechService(Protocol):
    def synthesize(
        self,
        text: str,
        language: str = "ko",
    ) -> Path:
        """
        입력 텍스트를 음성 파일로 변환한다.
        """
        ...
```

구현체 예시:

```text
BrowserTTSService
MacOSTTSService
PiperTTSService
KokoroTTSService
```

권장 프로젝트 구조:

```text
backend/
└── speech/
    ├── __init__.py
    ├── base.py
    ├── browser_tts.py
    ├── macos_tts.py
    ├── piper_tts.py
    ├── kokoro_tts.py
    └── text_normalizer.py
```

---

# 8. macOS TTS 서비스 예시

맥북에서 가장 간단하게 로컬 음성 파일을 생성하는 방식이다.

```python
from pathlib import Path
import subprocess
import uuid


class MacOSTTSService:
    def __init__(
        self,
        output_dir: str = "data/audio",
        voice: str = "Yuna",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.voice = voice

    def synthesize(
        self,
        text: str,
        language: str = "ko",
    ) -> Path:
        audio_id = str(uuid.uuid4())
        output_path = (
            self.output_dir
            / f"{audio_id}.aiff"
        )

        command = [
            "say",
            "-v",
            self.voice,
            "-o",
            str(output_path),
            text,
        ]

        subprocess.run(
            command,
            check=True,
        )

        return output_path
```

사용 예시:

```python
tts_service = MacOSTTSService()

audio_path = tts_service.synthesize(
    text="오늘 회의 일정은 오후 세 시입니다."
)
```

---

# 9. 음성 설정 테이블

사용자별 TTS 설정을 저장하려면 별도 설정 테이블을 둔다.

```sql
CREATE TABLE user_preferences (
    user_id UUID PRIMARY KEY,

    tts_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    tts_autoplay BOOLEAN NOT NULL DEFAULT FALSE,

    tts_engine VARCHAR(30)
        NOT NULL
        DEFAULT 'BROWSER',

    tts_voice VARCHAR(200),

    tts_language VARCHAR(20)
        NOT NULL
        DEFAULT 'ko-KR',

    tts_rate NUMERIC(4, 2)
        NOT NULL
        DEFAULT 1.0,

    created_at TIMESTAMPTZ
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMPTZ
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

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
        CHECK (
            tts_rate BETWEEN 0.5 AND 2.0
        )
);
```

---

# 10. 메시지별 오디오 저장 테이블

생성한 오디오 파일을 재사용하려면 메시지와 오디오를 연결하는 테이블을 추가할 수 있다.

```sql
CREATE TABLE message_audio (
    id UUID PRIMARY KEY
        DEFAULT gen_random_uuid(),

    message_id UUID NOT NULL,

    engine VARCHAR(30) NOT NULL,

    voice VARCHAR(200),

    language VARCHAR(20),

    audio_path TEXT NOT NULL,

    created_at TIMESTAMPTZ
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_message_audio_message
        FOREIGN KEY (message_id)
        REFERENCES chat_messages(id)
        ON DELETE CASCADE
);
```

초기 버전에서는 반드시 필요하지 않다.

처음에는 오디오 파일을 임시로 생성하고 재생해도 충분하다.

---

# 11. 음성 파일 저장 구조

오디오 파일을 저장한다면 다음 구조를 권장한다.

```text
data/
└── audio/
    └── {message_id}/
        └── response.wav
```

또는 UUID 기반으로 저장할 수 있다.

```text
data/
└── audio/
    ├── 1f8fa213-....wav
    └── 72dcb510-....wav
```

---

# 12. 텍스트 정규화

TTS에 전달하기 전에 텍스트를 정리하는 과정이 필요하다.

예를 들어 다음 내용을 제거하거나 변환해야 한다.

```text
Markdown 기호
코드 블록
URL
HTML 태그
표
특수문자
이모지
긴 파일 경로
```

예시:

````python
import re


def normalize_for_tts(text: str) -> str:
    normalized = text

    normalized = re.sub(
        r"```.*?```",
        "코드 블록이 포함되어 있습니다.",
        normalized,
        flags=re.DOTALL,
    )

    normalized = re.sub(
        r"`([^`]*)`",
        r"\1",
        normalized,
    )

    normalized = re.sub(
        r"https?://\S+",
        "링크",
        normalized,
    )

    normalized = re.sub(
        r"[*_>#-]",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()
````

---

# 13. 긴 응답 처리

긴 응답을 한 번에 읽으면 사용성이 떨어질 수 있다.

따라서 다음 방식 중 하나를 사용할 수 있다.

```text
전체 답변 읽기
요약만 읽기
첫 문단만 읽기
문단별 재생
사용자가 선택한 부분만 읽기
```

권장 초기 정책:

```text
응답이 짧으면 전체 읽기
응답이 길면 요약 읽기
코드 블록은 읽지 않기
출처 목록은 생략 가능
```

예시 기준:

```text
500자 이하
  전체 읽기

500자 초과
  요약 또는 첫 500자 읽기
```

---

# 14. 스트리밍 응답과 음성 재생

Gemma의 텍스트 응답은 스트리밍으로 표시할 수 있다.

하지만 TTS는 일반적으로 전체 문장이 완성된 뒤 생성하는 것이 안정적이다.

```text
텍스트
  실시간 스트리밍 표시

음성
  응답 완료 후 생성
```

고급 구현에서는 문장 단위로 TTS를 생성할 수 있다.

```text
첫 번째 문장 생성
   ↓
TTS 생성 및 재생

두 번째 문장 생성
   ↓
TTS 생성 및 재생
```

하지만 초기 버전에서는 구현 복잡도가 높으므로 권장하지 않는다.

---

# 15. 최종 권장 구현 방식

## 1차 버전

```text
텍스트 응답
  Gemma 3 4B

텍스트 표시
  Streamlit st.chat_message

음성 읽기
  브라우저 Web Speech API

UI
  응답별 🔊 읽기 버튼

옵션
  자동 읽기 ON/OFF
```

## 2차 버전

```text
로컬 TTS 엔진 추가

후보
  macOS say
  Piper
  Kokoro
```

## 3차 버전

```text
음성 질문 기능 추가

마이크
  ↓
STT
  ↓
Gemma + RAG
  ↓
텍스트 응답
  ↓
TTS
  ↓
음성 출력
```

---

# 16. 음성 입력 확장

완전한 음성 비서를 만들려면 STT 기능을 추가한다.

```text
사용자 음성
   ↓
마이크 입력
   ↓
Speech To Text
   ↓
사용자 질문 텍스트
   ↓
LangGraph
   ↓
Gemma + RAG
   ↓
텍스트 응답
   ↓
Text To Speech
   ↓
음성 출력
```

후보 STT 엔진:

```text
Whisper
faster-whisper
whisper.cpp
macOS Speech Recognition
```

M1 16GB 환경에서는 `whisper.cpp` 또는 `faster-whisper` 기반의 소형 모델부터 시작하는 것이 적절하다.

---

# 17. 최종 구조

```text
Chat Application
   ├─ 텍스트 입력
   ├─ 음성 입력
   ├─ 텍스트 응답
   └─ 음성 재생
          │
          ▼
Common Backend
   ├─ Chat Service
   ├─ RAG Service
   ├─ STT Service
   ├─ TTS Service
   └─ LangGraph
          │
          ├─ Ollama Gemma 3 4B
          ├─ PostgreSQL + pgvector
          └─ Local TTS / STT Engine
```

---

# 18. 최종 결정

초기 구현은 다음 구성을 권장한다.

```text
텍스트 생성
  Ollama + gemma3:4b

텍스트 표시
  Streamlit

음성 읽기
  Browser Web Speech API

대안
  macOS say

향후 확장
  Local TTS
  STT
  완전한 음성 비서
```

가장 먼저 구현할 기능은 다음과 같다.

```text
1. 챗봇 응답 텍스트 표시
2. 응답별 읽기 버튼
3. 자동 읽기 설정
4. 브라우저 또는 macOS 음성 합성
5. 긴 답변 텍스트 정규화
```
