"""
Admin Application — 지식베이스 관리
실행: streamlit run apps/admin_app.py --server.port 8502
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from backend.services import document_service, ingestion_service, session_service

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Base Admin",
    page_icon="📚",
    layout="wide",
)

# ──────────────────────────────────────────────
# 사용자 ID
# ──────────────────────────────────────────────
if "admin_user_id" not in st.session_state:
    try:
        st.session_state.admin_user_id = session_service.get_or_create_admin_user_id()
    except Exception:
        st.session_state.admin_user_id = None

USER_ID = st.session_state.get("admin_user_id")

# ──────────────────────────────────────────────
# 헤더
# ──────────────────────────────────────────────
st.title("📚 Knowledge Base Admin")

if USER_ID is None:
    st.error("DB 연결 실패. docker-compose를 확인하세요.")
    st.stop()

# ──────────────────────────────────────────────
# 탭 레이아웃
# ──────────────────────────────────────────────
tab_upload, tab_docs, tab_chunks = st.tabs(["📤 문서 업로드", "📋 문서 목록", "🔍 청크 미리보기"])


# ══════════════════════════════════════════════
# 탭 1: 문서 업로드
# ══════════════════════════════════════════════
with tab_upload:
    st.subheader("문서 업로드 및 벡터화")

    uploaded_file = st.file_uploader(
        "파일 선택",
        type=["pdf", "txt", "md", "docx", "png", "jpg", "jpeg"],
    )

    col1, col2 = st.columns(2)
    with col1:
        category = st.text_input("카테고리 (선택)", placeholder="예: SAC, Job Monitor, AI")
    with col2:
        scope = st.selectbox("범위", ["GLOBAL", "SESSION"])

    col3, col4 = st.columns(2)
    with col3:
        chunk_size = st.number_input(
            "청크 크기 (토큰)",
            min_value=50,
            max_value=2000,
            value=300,
            step=50,
        )
    with col4:
        chunk_overlap = st.number_input(
            "청크 오버랩",
            min_value=0,
            max_value=500,
            value=80,
            step=10,
        )

    if st.button("📥 업로드 및 벡터화 시작", disabled=uploaded_file is None):
        with st.spinner("처리 중..."):
            try:
                data = uploaded_file.read()
                mime_type = uploaded_file.type

                doc_id = document_service.register_document(
                    uploaded_by=USER_ID,
                    filename=uploaded_file.name,
                    data=data,
                    mime_type=mime_type,
                    category=category or None,
                    scope=scope,
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                )

                st.info(f"문서 등록 완료: `{doc_id}`\n\n임베딩 처리를 시작합니다...")

                # 동기 처리 (Streamlit 환경에서 단순하게 처리)
                ingestion_service.ingest_document(doc_id)

                st.success(f"✅ '{uploaded_file.name}' 처리 완료!")

            except Exception as e:
                st.error(f"처리 중 오류 발생: {e}")


# ══════════════════════════════════════════════
# 탭 2: 문서 목록
# ══════════════════════════════════════════════
with tab_docs:
    st.subheader("등록된 문서 목록")

    if st.button("🔄 새로고침", key="refresh_docs"):
        st.rerun()

    try:
        docs = document_service.list_documents()
    except Exception as e:
        st.error(f"문서 목록 로드 실패: {e}")
        docs = []

    if not docs:
        st.caption("등록된 문서가 없습니다.")
    else:
        _status_colors = {
            "READY": "🟢",
            "FAILED": "🔴",
            "UPLOADED": "🟡",
            "EXTRACTING": "🔵",
            "EXTRACTED": "🔵",
            "CHUNKING": "🔵",
            "CHUNKED": "🔵",
            "EMBEDDING": "🔵",
            "INACTIVE": "⚫",
            "DELETED": "⚫",
        }

        for doc in docs:
            status_icon = _status_colors.get(doc.status, "⚪")
            active_tag = "✅" if doc.is_active else "❌"

            with st.expander(
                f"{status_icon} {doc.file_name}  |  {doc.status}  |  청크 {doc.chunk_count}개",
                expanded=False,
            ):
                col_info, col_actions = st.columns([3, 1])

                with col_info:
                    st.markdown(f"**ID:** `{doc.id}`")
                    st.markdown(f"**카테고리:** {doc.category or '-'}")
                    st.markdown(f"**범위:** {doc.scope}")
                    st.markdown(f"**활성:** {active_tag}")
                    st.markdown(f"**크기:** {doc.file_size_bytes or '-'} bytes")
                    st.markdown(f"**페이지:** {doc.page_count or '-'}")
                    st.markdown(f"**청크:** {doc.chunk_count}")
                    st.markdown(f"**임베딩 모델:** {doc.embedding_model or '-'}")
                    st.markdown(f"**처리 완료:** {doc.processed_at or '-'}")
                    if doc.error_message:
                        st.error(f"오류: {doc.error_message}")

                with col_actions:
                    # 재처리
                    if st.button("🔄 재처리", key=f"reprocess_{doc.id}"):
                        with st.spinner("재처리 중..."):
                            try:
                                ingestion_service.reprocess_document(doc.id)
                                st.success("재처리 완료")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

                    # 활성화/비활성화 토글
                    if doc.is_active:
                        if st.button("⏸ 비활성화", key=f"deact_{doc.id}"):
                            document_service.deactivate_document(doc.id)
                            st.rerun()
                    else:
                        if st.button("▶ 활성화", key=f"act_{doc.id}"):
                            document_service.reactivate_document(doc.id)
                            st.rerun()

                    # 삭제
                    if st.button("🗑 삭제", key=f"delete_{doc.id}"):
                        document_service.delete_document(doc.id)
                        st.success("삭제 완료")
                        st.rerun()


# ══════════════════════════════════════════════
# 탭 3: 청크 미리보기
# ══════════════════════════════════════════════
with tab_chunks:
    st.subheader("문서 청크 미리보기")

    try:
        docs_for_preview = document_service.list_documents()
    except Exception:
        docs_for_preview = []

    if not docs_for_preview:
        st.caption("문서가 없습니다.")
    else:
        doc_options = {f"{d.file_name} ({d.id[:8]}...)": d.id for d in docs_for_preview}
        selected_label = st.selectbox("문서 선택", list(doc_options.keys()))
        selected_doc_id = doc_options[selected_label]

        page_size = st.number_input("페이지당 청크 수", min_value=5, max_value=50, value=10)
        page_num = st.number_input("페이지", min_value=1, value=1)
        offset = (page_num - 1) * int(page_size)

        try:
            from backend.database.repositories.chunk_repository import ChunkRepository
            chunk_repo = ChunkRepository()
            chunks = chunk_repo.list_by_document(
                document_id=selected_doc_id,
                page_size=int(page_size),
                offset=offset,
            )
        except Exception as e:
            st.error(f"청크 로드 실패: {e}")
            chunks = []

        if not chunks:
            st.caption("청크가 없습니다.")
        else:
            for c in chunks:
                with st.expander(
                    f"청크 {c.chunk_index}  |  페이지 {c.page_number or '-'}"
                ):
                    st.text(c.content)
                    if c.token_count:
                        st.caption(f"토큰 수: {c.token_count}")
