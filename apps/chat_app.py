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
from backend.services.chat_service import (
    STATUS_PREFIX as _STATUS_PREFIX,
    THINK_PREFIX as _THINK_PREFIX,
    FOLLOWUP_PREFIX as _FOLLOWUP_PREFIX,
    PLAN_PREFIX as _PLAN_PREFIX,
    _AGENT_DISPLAY as _AGENT_DISPLAY,
)
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
    st.markdown("# ChatSELMA")

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
    for idx, msg in enumerate(history):
        is_last_msg = (idx == len(history) - 1)
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

                # ── 후속 작업 Yes/No 버튼 (마지막 AI 메시지에만 표시) ────────
                if is_last_msg:
                    pf_raw = st.session_state.get("pending_followup")
                    if pf_raw:
                        parts = pf_raw.split(":::", 1)
                        pf_question = parts[0].strip()
                        pf_action   = parts[1].strip() if len(parts) > 1 else pf_question
                        st.markdown(f"---\n💡 **{pf_question}**")
                        col_yes, col_no, _ = st.columns([1.5, 1, 4])
                        with col_yes:
                            if st.button("✅ 예, 해줘!", key="followup_yes", use_container_width=True):
                                st.session_state["auto_input"] = pf_action
                                st.session_state.pop("pending_followup", None)
                                st.rerun()
                        with col_no:
                            if st.button("❌ 아니요", key="followup_no", use_container_width=True):
                                st.session_state.pop("pending_followup", None)
                                st.rerun()

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
# auto_input: Yes 버튼 클릭 시 자동 전송할 메시지
_auto = st.session_state.pop("auto_input", None)
user_input = _auto or st.chat_input("메시지를 입력하세요...")

task_state: dict = {"tasks": [], "completed": set(), "current": 0, "revised": set()}

if user_input:
    # 새 입력 시작 → 이전 상태 초기화
    st.session_state.pop("_task_panel_md", None)
    st.session_state.pop("pending_followup", None)

    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=_AVATAR):
            # 렌더링 순서: [계획 카드] → [추론(CoT)] → [상태] → [응답]
            plan_box    = st.empty()   # orchestrator 작업 계획 카드
            think_box   = st.empty()   # <think> 블록이 오면 expander로 교체
            status_ph   = st.empty()   # 진행 상황 표시
            placeholder = st.empty()   # 실제 응답 내용

            # 초기 대기 상태 표시
            placeholder.markdown(
                "<span style='color:#888;font-style:italic;'>⏳ 생각 중...</span>",
                unsafe_allow_html=True,
            )
            collected = []
            _has_content = False
            _has_inline_status = False

            # ── Chain-of-Thought 상태 ──────────────────────────────────────
            thinking_buf: list[str] = []
            think_ph = None
            thinking_sealed = False

            try:
                for token in chat_service.stream_chat(
                    session_id=current_session_id,
                    user_message=user_input,
                ):
                    # ── 작업 계획 토큰 ───────────────────────────────────────
                    if token.startswith(_PLAN_PREFIX):
                        raw = token[len(_PLAN_PREFIX):]
                        lines_html = []
                        for line in raw.strip().splitlines():
                            parts = line.split("|", 2)
                            if len(parts) == 3:
                                tid, agent, desc = parts
                                label = _AGENT_DISPLAY.get(agent.strip(), f"💡 {agent}")
                                lines_html.append(
                                    f"<div style='margin:2px 0'>"
                                    f"<b>{tid}.</b> {label} &mdash; {desc.strip()}"
                                    f"</div>"
                                )
                        # 채팅 응답 내 계획 카드
                        if lines_html:
                            with plan_box.container():
                                st.markdown(
                                    f"<div style='background:#f0f4ff;border-left:3px solid #6c8ebf;"
                                    f"padding:8px 12px;border-radius:4px;font-size:0.87em;"
                                    f"margin-bottom:4px;'>"
                                    f"<div style='font-weight:600;margin-bottom:4px;'>🧩 실행 계획</div>"
                                    f"{''.join(lines_html)}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                        continue

                    # ── 추론 (CoT) 토큰 ─────────────────────────────────────
                    if token.startswith(_THINK_PREFIX):
                        t = token[len(_THINK_PREFIX):]
                        thinking_buf.append(t)
                        if think_ph is None:
                            with think_box.expander("🤔 추론 과정", expanded=True):
                                think_ph = st.empty()
                        think_ph.markdown(
                            "```\n" + "".join(thinking_buf) + "▌\n```"
                        )
                        continue

                    # ── FOLLOWUP 토큰: Yes/No 버튼용 질문 저장 ──────────────
                    if token.startswith(_FOLLOWUP_PREFIX):
                        raw_pf = token[len(_FOLLOWUP_PREFIX):]
                        q = raw_pf.split(":::", 1)[0].strip()
                        # 정크 필터: LLM이 플레이스홀더를 그대로 출력한 경우 무시
                        _pf_junk = ["사용자에게 보여줄", "실제_질문", "질문:::작업", "내용:::내용"]
                        if not any(j in q for j in _pf_junk) and 5 < len(q) < 80:
                            st.session_state["pending_followup"] = raw_pf
                        continue

                    # 추론 종료 후 첫 실제 토큰 → expander 완료 처리
                    if thinking_buf and not thinking_sealed:
                        thinking_sealed = True
                        if think_ph:
                            think_ph.markdown(
                                "```\n" + "".join(thinking_buf) + "\n```"
                            )

                    # ── 진행 상황 상태 토큰 ─────────────────────────────────
                    if token.startswith(_STATUS_PREFIX):
                        status_text = token[len(_STATUS_PREFIX):]
                        if not _has_content:
                            placeholder.markdown(
                                f"<span style='color:#888;font-style:italic;'>{status_text}</span>",
                                unsafe_allow_html=True,
                            )
                        else:
                            status_ph.markdown(
                                f"<small style='color:#888;'>*{status_text}*</small>",
                                unsafe_allow_html=True,
                            )
                            _has_inline_status = True
                        continue

                    # ── 실제 응답 토큰 ───────────────────────────────────────
                    if not _has_content:
                        _has_content = True
                    if _has_inline_status:
                        _has_inline_status = False
                        status_ph.empty()

                    collected.append(token)
                    current_text = "".join(collected)
                    placeholder.markdown(current_text + "▌")
                    if "📋" in current_text:
                        try:
                            _render_task_panel(sidebar_task_panel, current_text, task_state, is_streaming=True)
                        except Exception as _panel_err:
                            _log.warning("task panel update failed: %s", _panel_err)

                _raw = "".join(collected)
                # [FOLLOWUP] 잔여물 제거
                _raw = _re.sub(r"\[FOLLOWUP\].*?\[/FOLLOWUP\]", "", _raw, flags=_re.DOTALL)
                _raw = _re.sub(r"\[/?FOLLOWUP[^\]]*\]?", "", _raw)
                # 가짜 출처 번호 제거: (출처 1), (출처 1, 5) 등
                _raw = _re.sub(r"\s*\(출처\s*[\d,\s]+\)", "", _raw).strip()
                final_text = fix_markdown_spacing(strip_leaked_prompt(_raw))
                placeholder.markdown(final_text)
                status_ph.empty()
                if task_state["tasks"]:
                    current_text_final = "".join(collected)
                    try:
                        _render_task_panel(sidebar_task_panel, current_text_final, task_state, is_streaming=False)
                    except Exception as _panel_err:
                        _log.warning("task panel final update failed: %s", _panel_err)

            except Exception as e:
                placeholder.error(f"응답 생성 중 오류가 발생했습니다: {e}")

    st.rerun()

