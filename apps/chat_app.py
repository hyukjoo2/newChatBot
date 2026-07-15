"""
Chat Application
실행: streamlit run apps/chat_app.py --server.port 8501
"""
from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image as _Image

from backend.services import session_service, chat_service, document_service, ingestion_service
from backend.chatbot.language_utils import strip_leaked_prompt, fix_markdown_spacing

_ICON_PATH = Path(__file__).parent / "static" / "ttlzzang.png"
_AVATAR = _Image.open(_ICON_PATH)

import re as _re
import logging as _logging
_log = _logging.getLogger(__name__)


# ── 태스크 패널 헬퍼 ─────────────────────────────────────────────────────────

def _parse_tasks_from_text(text: str) -> list[dict]:
    """스트리밍 텍스트에서 📋 작업 목록 항목(및 이유)을 파싱한다."""
    tasks: list[dict] = []
    if "📋" not in text:
        return tasks
    in_list = False
    current: dict | None = None
    for line in text.split("\n"):
        if "📋" in line:
            in_list = True
            continue
        if not in_list:
            continue
        # 작업 번호 라인: "1. [ ] 내용" 또는 "1. 내용"
        m = _re.match(r"\s*(\d+)\.\s*(?:\[[ x]\])?\s*(.+)", line)
        if m:
            if current:
                tasks.append(current)
            current = {"id": int(m.group(1)), "desc": m.group(2).strip(), "reason": ""}
            continue
        # 이유 라인: "이유: ..." 또는 "   이유: ..."
        if current and _re.match(r"\s*이유\s*:", line):
            reason_text = _re.sub(r"^\s*이유\s*:\s*", "", line).strip()
            current["reason"] = reason_text
            continue
        # 빈 줄은 스트리밍 중 무시
        if not line.strip():
            continue
        # "위 순서대로 작업을 시작하겠습니다" 같은 종료 문장이면 목록 끝
        if current and not _re.match(r"\s*(\d+[\.\)]|\-|\*|이유)", line):
            tasks.append(current)
            current = None
            break
    if current:
        tasks.append(current)
    return tasks


def _render_task_panel(panel, text: str, state: dict, is_streaming: bool = True) -> None:
    """스트리밍 텍스트를 파싱해 사이드바 태스크 패널을 업데이트한다."""
    parsed = _parse_tasks_from_text(text)
    if parsed:
        state["tasks"] = parsed
    if not state["tasks"]:
        return

    # 완료/진행 중 상태 갱신
    # 방법 1: 명시적 ✅ 작업 N 마커
    for m in _re.finditer(r"✅[^\n]*?작업\s*(\d+)|작업\s*(\d+)[^\n]*?✅", text):
        tid = int(m.group(1) or m.group(2))
        state["completed"].add(tid)
    # 방법 2: ▶ 작업 N+1 이 나왔으면 이전 작업들은 묵시적 완료
    started: list[int] = []
    for m in _re.finditer(r"▶[^\n]*?작업\s*(\d+)", text):
        started.append(int(m.group(1)))
    if len(started) >= 2:
        for tid in started[:-1]:
            state["completed"].add(tid)
    state["current"] = started[-1] if started else 0

    # 적응적 계획 수정: 🔄 계획 수정/추가 마커 반영
    for m in _re.finditer(r'🔄\s*계획\s*수정\s*:\s*작업\s*(\d+)\s*[→\-]+\s*[""""]?([^""""\n(]+)', text):
        tid, new_desc = int(m.group(1)), m.group(2).strip().rstrip('"""".').strip()
        for t in state["tasks"]:
            if t["id"] == tid and tid not in state["completed"]:
                t["desc"] = new_desc
                state.setdefault("revised", set()).add(tid)
                break
    for m in _re.finditer(r'🔄\s*계획\s*추가\s*:\s*작업\s*(\d+)\s*[→\-]+\s*[""""]?([^""""\n(]+)', text):
        tid, desc = int(m.group(1)), m.group(2).strip().rstrip('"""".').strip()
        if not any(t["id"] == tid for t in state["tasks"]):
            state["tasks"].append({"id": tid, "desc": desc, "reason": ""})
            state["tasks"].sort(key=lambda t: t["id"])
            state.setdefault("revised", set()).add(tid)

    # 사이드바 네이티브 마크다운으로 렌더링
    lines = ["---", "**📋 작업 현황**", ""]
    for t in state["tasks"]:
        tid = t["id"]
        reason = t.get("reason", "")
        revised_ids = state.get("revised", set())
        if tid in state["completed"]:
            icon = "✅"
        elif tid in revised_ids and tid not in state["completed"]:
            icon = "🔄"
        elif tid == state["current"]:
            icon = "⏳"
        else:
            icon = "⬜"
        desc = t["desc"][:40] + ("…" if len(t["desc"]) > 40 else "")
        lines.append(f"{icon} **{tid}.** {desc}  ")   # 두 칸 공백 → 강제 줄바꿈
        if reason:
            short = reason[:55] + ("…" if len(reason) > 55 else "")
            lines.append(f"&nbsp;&nbsp;&nbsp;↳ *{short}*  ")
        lines.append("")   # 항목 간 빈 줄

    if is_streaming:
        lines.append("*🔄 AI가 작업하고 있습니다...*")

    md = "\n".join(lines)
    panel.markdown(md, unsafe_allow_html=True)
    # 완료 후에만 session_state에 보존 (다음 rerun 후 복원용)
    if not is_streaming:
        st.session_state["_task_panel_md"] = md


# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Selma",
    page_icon=_AVATAR,
    layout="wide",
)

# ──────────────────────────────────────────────
# 사용자 ID 초기화 (Admin 사용자 고정)
# ──────────────────────────────────────────────
if "user_id" not in st.session_state:
    try:
        st.session_state.user_id = session_service.get_or_create_admin_user_id()
    except Exception:
        st.session_state.user_id = None

USER_ID = st.session_state.get("user_id")

# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _load_sessions() -> list:
    if not USER_ID:
        return []
    try:
        return session_service.list_sessions(USER_ID)
    except Exception:
        return []


def _new_session() -> str:
    session_id = session_service.create_session(USER_ID, mode="CHAT")
    st.session_state.current_session_id = session_id
    return session_id


def _format_time(dt) -> str:
    if dt is None:
        return ""
    try:
        return dt.strftime("%m/%d %H:%M")
    except Exception:
        return str(dt)


# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────

with st.sidebar:
    # ── 태스크 진행 패널 (task_agent 실행 시에만 표시) ───────────────────────
    sidebar_task_panel = st.empty()
    # 페이지 재렌더링 후 이전 작업 목록 복원 (없으면 자동으로 빈 상태)
    _saved_md = st.session_state.get("_task_panel_md", "")
    if _saved_md:
        sidebar_task_panel.markdown(_saved_md, unsafe_allow_html=True)
    # 사이드바 버튼 좌측 정렬 CSS
    st.html("""
    <style>
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
        justify-content: flex-start !important;
        text-align: left !important;
    }
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]
        div[data-testid="stMarkdownContainer"] {
        width: 100% !important;
        text-align: left !important;
    }
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] p {
        text-align: left !important;
        width: 100% !important;
    }
    </style>
    """)
    st.markdown("# Selma")

    if USER_ID is None:
        st.error("DB 연결 실패. docker-compose를 확인하세요.")
        st.stop()

    # 새 대화 버튼
    if st.button("+ 새 대화", use_container_width=True):
        _new_session()
        st.rerun()

    st.divider()
    st.subheader("대화 목록")

    sessions = _load_sessions()

    if not sessions:
        st.caption("대화가 없습니다.")
    else:
        for session in sessions:
            col_btn, col_menu = st.columns([5, 1])
            with col_btn:
                label = session.title
                time_tag = _format_time(session.last_message_at or session.created_at)
                if st.button(
                    f"💬 {label}\n{time_tag}",
                    key=f"sess_{session.id}",
                    use_container_width=True,
                ):
                    st.session_state.current_session_id = session.id
                    st.rerun()
            with col_menu:
                with st.popover("⋯", use_container_width=True):
                    st.caption(f"**{session.title}**")
                    new_name = st.text_input(
                        "이름",
                        value=session.title,
                        key=f"rename_input_{session.id}",
                        label_visibility="collapsed",
                        placeholder="새 이름 입력",
                    )
                    if st.button("✏️ 이름 변경", key=f"rename_btn_{session.id}", use_container_width=True):
                        if new_name.strip():
                            session_service.rename_session(session.id, new_name.strip())
                            st.rerun()
                    st.divider()
                    if st.button("🗑 삭제", key=f"del_{session.id}", use_container_width=True, type="primary"):
                        session_service.delete_session(session.id)
                        if st.session_state.get("current_session_id") == session.id:
                            st.session_state.pop("current_session_id", None)
                        st.rerun()

    st.divider()
    with st.expander("🔧 시스템 진단"):
        if st.button("헬스체크 실행", use_container_width=True):
            from backend.services.health_service import run_health_check
            with st.spinner("점검 중..."):
                report = run_health_check()
            st.text(report.summary())
            if not report.healthy:
                st.error("일부 구성 요소에 문제가 있습니다.")

# ──────────────────────────────────────────────
# 메인 화면
# ──────────────────────────────────────────────

current_session_id: str | None = st.session_state.get("current_session_id")

if current_session_id is None:
    st.markdown("## 새 대화를 시작하세요")
    st.caption("왼쪽 사이드바에서 **+ 새 대화** 버튼을 눌러 시작하세요.")
    st.stop()

# 세션 정보 로드
try:
    session_obj = session_service.get_session(current_session_id)
except Exception:
    session_obj = None

if session_obj is None:
    st.warning("세션을 찾을 수 없습니다.")
    st.stop()

# 헤더
mode_label = "🔍 지식 검색 (RAG)" if session_obj.default_mode == "RAG" else "💬 "
st.subheader(f"{mode_label} {session_obj.title}")

st.divider()

# 대화 기록 표시
try:
    history = chat_service.get_history(current_session_id)
except Exception as e:
    st.error(f"대화 기록 로드 실패: {e}")
    history = []

chat_container = st.container()
with chat_container:
    for msg in history:
        if msg.role == "USER":
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif msg.role == "ASSISTANT":
            with st.chat_message("assistant", avatar=_AVATAR):
                st.markdown(fix_markdown_spacing(msg.content))
                # 출처 표시 (RAG 모드)
                sources = msg.metadata.get("sources") or []
                if sources:
                    with st.expander("📄 출처"):
                        for i, src in enumerate(sources, 1):
                            page_info = f", {src['page_number']}페이지" if src.get("page_number") else ""
                            col_name, col_dl = st.columns([5, 1])
                            with col_name:
                                st.markdown(f"**{i}.** {src['file_name']}{page_info}")
                            with col_dl:
                                try:
                                    from pathlib import Path
                                    from backend.config import settings as _cfg
                                    _doc_path = next(
                                        Path(_cfg.upload_dir).glob(
                                            f"{src['document_id']}/*"
                                        ),
                                        None,
                                    )
                                    if _doc_path and _doc_path.exists():
                                        st.download_button(
                                            "⬇",
                                            data=_doc_path.read_bytes(),
                                            file_name=src["file_name"],
                                            key=f"dl_{msg.id}_{i}",
                                            help="원본 파일 다운로드",
                                        )
                                except Exception:
                                    pass
                # TTS 컨트롤 제거됨

# ── + 메뉴 팝오버 (채팅 입력창 왼쪽) ──────────────
st.markdown("""
<style>
/* + 버튼을 채팅 입력창 왼쪽에 fixed 배치 */
div[data-testid="stPopover"]:has(button[title="파일 업로드"]) {
    position: fixed;
    bottom: 16px;
    left: calc(var(--sidebar-width, 15rem) + 1.5rem);
    z-index: 1000;
}
div[data-testid="stPopover"]:has(button[title="파일 업로드"]) > button {
    width: 40px !important;
    height: 40px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    font-size: 20px !important;
    line-height: 1 !important;
}
/* chat input에 왼쪽 여백 확보 */
[data-testid="stBottom"] [data-testid="stChatInput"] {
    padding-left: 3rem;
}
/* 사이드바 대화 목록 버튼 좌측 정렬 */
</style>
""", unsafe_allow_html=True)

with st.popover("➕", use_container_width=False, help="지식베이스에 추가"):
    st.markdown("#### 📎 지식베이스에 추가")
    _tab_file, _tab_text = st.tabs(["파일 업로드", "텍스트 붙여넣기"])

    # ── 탭1: 파일 업로드 ──────────────────────────────────────
    with _tab_file:
        _upload_file = st.file_uploader(
            "파일 선택 (PDF, TXT, MD, DOCX, 이미지)",
            type=["pdf", "txt", "md", "docx", "png", "jpg", "jpeg"],
            key="quick_file_upload",
            label_visibility="collapsed",
        )
        if _upload_file is not None:
            _col_cat, _col_scope = st.columns(2)
            with _col_cat:
                _upload_category = st.text_input(
                    "카테고리",
                    placeholder="예: 매뉴얼, 정책",
                    key="quick_upload_category",
                    label_visibility="visible",
                ) or None
            with _col_scope:
                _upload_scope = st.selectbox(
                    "범위",
                    ["GLOBAL", "SESSION"],
                    key="quick_upload_scope",
                )
            if st.button("📥 지식베이스에 추가", use_container_width=True, key="quick_upload_btn"):
                with st.spinner(f"'{_upload_file.name}' 처리 중..."):
                    try:
                        _data = _upload_file.read()
                        _doc_id = document_service.register_document(
                            uploaded_by=USER_ID,
                            filename=_upload_file.name,
                            data=_data,
                            mime_type=_upload_file.type,
                            category=_upload_category,
                            scope=_upload_scope,
                            session_id=current_session_id if _upload_scope == "SESSION" else None,
                        )
                        _existing_doc = document_service.get_document(_doc_id)
                        if _existing_doc is None or _existing_doc.status != "READY":
                            ingestion_service.ingest_document(_doc_id)
                        if _upload_scope == "SESSION":
                            chat_service.inject_session_document_context(
                                session_id=current_session_id,
                                file_name=_upload_file.name,
                                document_id=_doc_id,
                            )
                        st.success(f"✅ '{_upload_file.name}' 추가 완료!")
                    except Exception as _e:
                        st.error(f"처리 오류: {_e}")

    # ── 탭2: 텍스트 붙여넣기 ──────────────────────────────────
    with _tab_text:
        _txt_title = st.text_input(
            "제목 (파일명으로 사용)",
            placeholder="예: 회의록_20260712",
            key="paste_title",
        )
        _txt_content = st.text_area(
            "내용",
            placeholder="텍스트를 붙여넣으세요...",
            height=200,
            key="paste_content",
        )
        _col_pcat, _col_pscope = st.columns(2)
        with _col_pcat:
            _paste_category = st.text_input(
                "카테고리",
                placeholder="예: 회의록",
                key="paste_category",
            ) or None
        with _col_pscope:
            _paste_scope = st.selectbox(
                "범위",
                ["GLOBAL", "SESSION"],
                key="paste_scope",
            )
        if st.button("📥 지식베이스에 추가", use_container_width=True, key="quick_paste_btn"):
            if not _txt_title.strip():
                st.warning("제목을 입력해 주세요.")
            elif not _txt_content.strip():
                st.warning("내용을 입력해 주세요.")
            else:
                _paste_filename = _txt_title.strip()
                if not _paste_filename.endswith(".txt"):
                    _paste_filename += ".txt"
                with st.spinner(f"'{_paste_filename}' 처리 중..."):
                    try:
                        _paste_data = _txt_content.encode("utf-8")
                        _doc_id = document_service.register_document(
                            uploaded_by=USER_ID,
                            filename=_paste_filename,
                            data=_paste_data,
                            mime_type="text/plain",
                            category=_paste_category,
                            scope=_paste_scope,
                            session_id=current_session_id if _paste_scope == "SESSION" else None,
                        )
                        ingestion_service.ingest_document(_doc_id)
                        if _paste_scope == "SESSION":
                            chat_service.inject_session_document_context(
                                session_id=current_session_id,
                                file_name=_paste_filename,
                                document_id=_doc_id,
                            )
                        st.success(f"✅ '{_paste_filename}' 추가 완료!")
                    except Exception as _e:
                        st.error(f"처리 오류: {_e}")

# 메시지 입력
user_input = st.chat_input("메시지를 입력하세요...")

task_state: dict = {"tasks": [], "completed": set(), "current": 0, "revised": set()}

if user_input:
    # 새 입력이 시작되면 이전 작업 목록 삭제 (sidebar_task_panel은 다음 rerender 시 비워짐)
    st.session_state.pop("_task_panel_md", None)
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=_AVATAR):
            placeholder = st.empty()
            placeholder.markdown(
                "<span style='color:#888;font-style:italic;'>⏳ 생각 중...</span>",
                unsafe_allow_html=True,
            )
            collected = []

            try:
                for token in chat_service.stream_chat(
                    session_id=current_session_id,
                    user_message=user_input,
                ):
                    collected.append(token)
                    current_text = "".join(collected)
                    placeholder.markdown(current_text + "▌")
                    # 📋 이후에는 매 토큰마다 패널 갱신
                    # panel 업데이트 실패가 스트리밍 루프를 끊으면 응답이 저장 안 되므로 격리
                    if "📋" in current_text:
                        try:
                            _render_task_panel(sidebar_task_panel, current_text, task_state, is_streaming=True)
                        except Exception as _panel_err:
                            _log.warning("task panel update failed: %s", _panel_err)

                final_text = fix_markdown_spacing(strip_leaked_prompt("".join(collected)))
                placeholder.markdown(final_text)
                # 완료 후: 스피너 없는 완료 상태로 패널 갱신 후 session_state에 저장
                if task_state["tasks"]:
                    current_text_final = "".join(collected)
                    try:
                        _render_task_panel(sidebar_task_panel, current_text_final, task_state, is_streaming=False)
                    except Exception as _panel_err:
                        _log.warning("task panel final update failed: %s", _panel_err)

            except Exception as e:
                placeholder.error(f"응답 생성 중 오류가 발생했습니다: {e}")

    st.rerun()

