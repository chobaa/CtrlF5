import streamlit as st
import time
import os
import streamlit.components.v1 as components # [New] JS 실행을 위한 모듈
from dotenv import load_dotenv

# 내부 모듈
from core.database import load_vectorstore, fetch_url_content, build_vectorstore, save_config, load_config
from core.engine import get_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# [CSS 스타일링] 모던 다크 그레이 & 미니멀
st.set_page_config(page_title="Ctrl + F5: DevRAG", page_icon="⌨️", layout="wide")
st.markdown("""
<style>
    /* 1. 메인 영역 스타일 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* 2. 채팅 메시지 박스 (심플 & 플랫) */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        box-shadow: none;
        padding: 15px;
    }
    
    /* 3. 사이드바 스타일 (깔끔한 회색톤) */
    [data-testid="stSidebar"] {
        background-color: #fafafa;
        border-right: 1px solid #f4f4f5;
    }
    [data-testid="stSidebar"] h1 {
        margin-bottom: 0px;
    }
    [data-testid="stSidebar"] h3 {
        color: #71717a;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 20px;
    }
    
    /* 4. 입력창 스타일 (미니멀) */
    .stTextArea textarea {
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 14px;
        line-height: 1.5;
        background-color: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 8px;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 1px #3b82f6;
    }
    
    /* 5. 버튼 스타일 (블루 포인트) */
    button[kind="primary"] {
        background-color: #2563eb;
        border: none;
        font-weight: 600;
        border-radius: 6px;
        transition: all 0.2s;
    }
    button[kind="primary"]:hover {
        background-color: #1d4ed8;
    }
    
    /* 일반 버튼 호버 효과 */
    button[kind="secondary"] {
        border-color: #e4e4e7;
        color: #3f3f46;
    }
    button[kind="secondary"]:hover {
        border-color: #2563eb;
        color: #2563eb;
        background-color: #eff6ff;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. 상태 및 설정 로드 ---
if "stacks" not in st.session_state:
    st.session_state.stacks = load_config()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. 콜백 함수 ---
def add_stack_callback():
    new_stack = st.session_state.new_stack_input.strip()
    if new_stack and new_stack not in st.session_state.stacks:
        st.session_state.stacks[new_stack] = []
        save_config(st.session_state.stacks)
        st.toast(f"✅ '{new_stack}' 스택이 추가되었습니다!", icon="🎉")
    st.session_state.new_stack_input = ""

def add_url_callback():
    new_url = st.session_state.new_url_input.strip()
    current_stack = st.session_state.current_stack_selection
    if new_url:
        if new_url not in st.session_state.stacks[current_stack]:
            st.session_state.stacks[current_stack].append(new_url)
            save_config(st.session_state.stacks)
            st.toast("✅ URL이 추가되었습니다!", icon="🔗")
        else:
            st.toast("⚠️ 이미 존재하는 URL입니다.", icon="✋")
    st.session_state.new_url_input = ""

def append_to_prompt(text):
    """단축 버튼 콜백"""
    if "draft_message" in st.session_state and st.session_state.draft_message:
        st.session_state.draft_message += f"\n\n{text}"
    else:
        st.session_state.draft_message = text

def send_message_callback():
    """전송 버튼 콜백"""
    if "draft_message" in st.session_state and st.session_state.draft_message.strip():
        user_msg = st.session_state.draft_message
        st.session_state.messages.append({"role": "user", "content": user_msg})
        st.session_state.draft_message = ""

# --- 3. 사이드바 UI ---
with st.sidebar:
    st.markdown("""
        <h1 style="font-family: 'Inter', sans-serif; font-weight: 800; color: #18181b; font-size: 24px; margin-bottom: 10px;">
            Ctrl + F5 <span style="font-weight: 500; color: #a1a1aa; font-size: 16px; margin-left: 4px;">Admin</span>
        </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ 설정")
    strict_mode = st.toggle("🛡️ 엄격 모드", value=True, 
                            help="켜기: 문서에 있는 내용만 대답합니다.\n끄기: AI의 일반 지식도 함께 사용합니다.")
    
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

# --- 4. 메인 영역 UI ---
c1, c2 = st.columns([0.7, 0.3])
with c1:
    st.markdown(f"""
        <h2 style="font-family: 'Inter', sans-serif; font-weight: 700; color: #18181b; font-size: 30px; display: flex; align-items: center; margin-bottom: 0;">
            ⌨️ Ctrl + F5
            <span style="color: #e4e4e7; margin: 0 12px; font-weight: 300;">|</span>
            <span style="background-color: #f4f4f5; color: #18181b; padding: 4px 14px; border-radius: 8px; font-size: 24px; font-weight: 600;">
                {selected_stack}
            </span>
        </h2>
    """, unsafe_allow_html=True)
with c2:
    if strict_mode:
        st.markdown("""<div style="text-align: right; padding-top: 15px;"><span style="background-color: #fef2f2; color: #991b1b; padding: 6px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; display: inline-block; border: 1px solid #fecaca;">🛡️ 엄격 모드 ON</span></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="text-align: right; padding-top: 15px;"><span style="background-color: #eff6ff; color: #1e40af; padding: 6px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; display: inline-block; border: 1px solid #dbeafe;">🧠 일반 모드</span></div>""", unsafe_allow_html=True)

st.caption("최신 공식 문서를 기반으로 질문하고, 확신을 가지고 코딩하세요.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ .env 파일에 OPENAI_API_KEY를 설정해주세요.")
    st.stop()

vectorstore = load_vectorstore(selected_stack)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if "제공된 문서에서 해당 내용에 대한 근거를 찾을 수 없습니다" in m["content"]:
             st.warning(m["content"])
        else:
             st.markdown(m["content"])

# --- 5. RAG 답변 생성 ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_msg = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
        with st.spinner("최신 문서 분석 중..."):
            try:
                rag_chain = get_rag_chain(vectorstore, selected_stack, is_strict=strict_mode)
                response = rag_chain.invoke({"input": last_user_msg, "chat_history": history})
                answer_text = response['answer']
                
                if strict_mode and "제공된 문서에서 해당 내용에 대한 근거를 찾을 수 없습니다" in answer_text:
                    st.warning("⚠️ **정보 부족 (Strict Mode)**\n\n문서에서 근거를 찾지 못했습니다.")
                    st.session_state.messages.append({"role": "assistant", "content": answer_text})
                else:
                    st.markdown(answer_text)
                    st.session_state.messages.append({"role": "assistant", "content": answer_text})
                    with st.expander("🔍 참조 문서 (Source)"):
                        if response.get('context'):
                            for i, doc in enumerate(response['context']):
                                st.markdown(f"**🔗 출처 {i+1}:** `{doc.metadata.get('source', '알 수 없음')}`")
                                st.caption(doc.page_content[:250].replace("\n", " ") + "...")
                                st.divider()
                        elif not strict_mode:
                            st.info("일반 지식 활용")
            except Exception as e:
                st.error(f"오류 발생: {e}")

# --- 6. 입력 영역 ---
st.divider()

# 6-1. 메시지 작성 (Text Area)
with st.container():
    prompt_text = st.text_area(
        "메시지 작성", 
        key="draft_message", 
        height=150,
        placeholder="여기에 코드를 붙여넣거나 질문을 입력하세요...\n(전송: Ctrl + Enter)",
        label_visibility="collapsed"
    )

# 6-2. 버튼 그룹
col_act1, col_act2, col_act3, col_send = st.columns([0.2, 0.2, 0.3, 0.3])

col_act1.button("🔍 코드 리뷰", use_container_width=True, 
                on_click=append_to_prompt, args=("위 코드의 잠재적인 버그와 개선점을 리뷰해줘.",))

col_act2.button("🐛 버그 찾기", use_container_width=True, 
                on_click=append_to_prompt, args=("위 코드에서 발생하는 에러의 원인과 해결책을 알려줘.",))

col_act3.button("📖 로직 설명", use_container_width=True, 
                on_click=append_to_prompt, args=("위 코드의 로직과 동작 원리를 상세히 설명해줘.",))

col_send.button("전송 🚀", type="primary", use_container_width=True, on_click=send_message_callback)

# --- 7. Ctrl + Enter 전송을 위한 JS 코드 ---
# 이 스크립트는 화면에 보이지 않으며, Ctrl+Enter 감지 시 '전송' 버튼을 클릭합니다.
components.html("""
<script>
document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const buttons = Array.from(window.parent.document.querySelectorAll('button[kind="primary"]'));
        const sendButton = buttons.find(el => el.innerText.includes('전송'));
        if (sendButton) {
            sendButton.click();
            e.preventDefault();
        }
    }
});
</script>
""", height=0, width=0)
