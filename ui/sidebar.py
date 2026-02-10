import streamlit as st
import time
from core.database import fetch_url_content, build_vectorstore, save_config
from core.callbacks import add_stack_callback, add_url_callback

def render_sidebar():
    with st.sidebar:
        st.markdown("""
            <h1 style="font-family: 'Inter', sans-serif; font-weight: 800; color: #18181b; font-size: 24px; margin-bottom: 10px;">
                Ctrl + F5 <span style="font-weight: 500; color: #a1a1aa; font-size: 16px; margin-left: 4px;">Admin</span>
            </h1>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ 설정")
        strict_mode = st.toggle("🛡️ 엄격 모드", value=True, 
                                help="켜기: 문서에 있는 내용만 대답합니다.\n끄기: AI의 일반 지식도 함께 사용합니다.")
        
        st.caption("참조 문서 개수 (Top-K)")
        top_k = st.slider("참조 문서 개수", min_value=1, max_value=20, value=5, label_visibility="collapsed",
                          help="최종적으로 LLM에 전달할 문서의 최대 개수입니다.")
        
        st.divider()

        st.markdown("### 🛠️ 기술 스택 관리")
        stack_list = list(st.session_state.stacks.keys())
        idx = 0
        if "current_stack_selection" in st.session_state and st.session_state.current_stack_selection in stack_list:
            idx = stack_list.index(st.session_state.current_stack_selection)

        selected_stack = st.selectbox("스택 선택", stack_list, index=idx, key="current_stack_selection", label_visibility="collapsed")
        
        with st.expander("➕ 새 스택 추가"):
            st.text_input("스택 이름", key="new_stack_input", on_change=add_stack_callback)

        st.divider()

        st.markdown(f"### 📚 지식 베이스")
        st.caption(f"**{selected_stack}** 학습 문서 관리")
        st.text_input("URL 추가", placeholder="https://docs...", key="new_url_input", on_change=add_url_callback, label_visibility="collapsed")

        urls = st.session_state.stacks.get(selected_stack, [])
        if urls:
            with st.container(border=True):
                for i, url in enumerate(urls):
                    c1, c2 = st.columns([0.75, 0.25])
                    c1.text(f"{url[:20]}...")
                    if c2.button("🗑️", key=f"del_{i}", help="삭제", use_container_width=True):
                        st.session_state.stacks[selected_stack].pop(i)
                        save_config(st.session_state.stacks)
                        st.rerun()
            st.caption(f"총 {len(urls)}개의 문서")
        else:
            st.info("등록된 문서가 없습니다.")

        st.divider()

        if st.button("🔄 RAG 엔진 업데이트", type="primary", use_container_width=True):
            if not urls:
                st.warning("URL을 먼저 추가해주세요.")
            else:
                with st.status(f"🚀 '{selected_stack}' 엔진 구축 중...", expanded=True) as status:
                    all_docs = []
                    start_time = time.time()
                    bar = st.progress(0)
                    for i, url in enumerate(urls):
                        st.write(f"📥 수집 중: `{url}`")
                        docs = fetch_url_content(url)
                        if docs: all_docs.extend(docs)
                        bar.progress((i + 1) / len(urls))
                        time.sleep(0.05)
                    if all_docs:
                        build_vectorstore(all_docs, selected_stack)
                        elapsed = time.time() - start_time
                        status.update(label=f"✅ 완료! ({elapsed:.1f}초)", state="complete", expanded=False)
                        st.success("업데이트 완료!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        status.update(label="❌ 실패", state="error")
                        st.error("문서 내용을 찾을 수 없습니다.")

        return selected_stack, strict_mode, top_k
