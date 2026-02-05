import streamlit as st
import time
import os
from dotenv import load_dotenv

# 내부 모듈
from core.database import load_vectorstore, fetch_url_content, build_vectorstore, save_config, load_config
from core.engine import get_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
st.set_page_config(page_title="Ctrl + F5: DevRAG", page_icon="⌨️", layout="wide")

# --- 1. 상태 및 설정 로드 ---
if "stacks" not in st.session_state:
    st.session_state.stacks = load_config()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. 콜백 함수 (Enter 키 처리) ---
def add_stack_callback():
    new_stack = st.session_state.new_stack_input.strip()
    if new_stack and new_stack not in st.session_state.stacks:
        st.session_state.stacks[new_stack] = []
        save_config(st.session_state.stacks)
        st.toast(f"✅ '{new_stack}' 스택 추가됨!", icon="🎉")
    st.session_state.new_stack_input = "" # 입력창 초기화

def add_url_callback():
    new_url = st.session_state.new_url_input.strip()
    current_stack = st.session_state.current_stack_selection
    if new_url:
        if new_url not in st.session_state.stacks[current_stack]:
            st.session_state.stacks[current_stack].append(new_url)
            save_config(st.session_state.stacks)
            st.toast("✅ URL 추가 완료!", icon="🔗")
        else:
            st.toast("⚠️ 이미 존재하는 URL입니다.", icon="✋")
    st.session_state.new_url_input = "" # 입력창 초기화

# --- 3. 사이드바 UI ---
with st.sidebar:
    st.title("⌨️ Ctrl + F5 Admin")
    
    # 스택 선택 및 추가
    st.subheader("🛠️ 스택 관리")
    stack_list = list(st.session_state.stacks.keys())
    
    # 세션 상태 동기화를 위해 index 계산
    idx = 0
    if "current_stack_selection" in st.session_state and st.session_state.current_stack_selection in stack_list:
        idx = stack_list.index(st.session_state.current_stack_selection)

    selected_stack = st.selectbox(
        "현재 작업 스택", 
        stack_list, 
        index=idx,
        key="current_stack_selection"
    )
    
    with st.expander("➕ 새 스택 추가 (Enter)"):
        st.text_input("스택 이름 입력", key="new_stack_input", on_change=add_stack_callback)

    st.divider()

    # URL 관리
    st.subheader(f"📚 {selected_stack} 지식 관리")
    st.text_input("URL 입력 후 Enter", placeholder="https://docs...", key="new_url_input", on_change=add_url_callback)

    # URL 리스트 표시
    urls = st.session_state.stacks.get(selected_stack, [])
    if urls:
        st.caption(f"총 {len(urls)}개의 문서 대기 중")
        for i, url in enumerate(urls):
            c1, c2 = st.columns([0.85, 0.15])
            c1.text(f"📄 {url[:30]}...")
            if c2.button("✕", key=f"del_{i}"):
                st.session_state.stacks[selected_stack].pop(i)
                save_config(st.session_state.stacks)
                st.rerun()
    else:
        st.info("URL을 추가해주세요.")

    st.divider()

    # 🔥 RAG 생성 (하이라이트 기능)
    if st.button("🔥 RAG 엔진 생성/갱신", type="primary"):
        if not urls:
            st.warning("URL을 먼저 추가하세요.")
        else:
            # 진행 상황 시각화 (Status Container)
            with st.status(f"🚀 '{selected_stack}' 지식 베이스 구축 중...", expanded=True) as status:
                all_docs = []
                start_time = time.time()
                
                # Progress Bar
                bar = st.progress(0)
                
                for i, url in enumerate(urls):
                    st.write(f"📡 Scraping: `{url}`")
                    docs = fetch_url_content(url)
                    if docs: all_docs.extend(docs)
                    
                    # 진행률 업데이트
                    bar.progress((i + 1) / len(urls))
                    time.sleep(0.05) # 시각적 효과

                if all_docs:
                    st.write(f"✂️ Chunking & Embedding ({len(all_docs)} docs)...")
                    build_vectorstore(all_docs, selected_stack)
                    
                    elapsed = time.time() - start_time
                    status.update(label=f"✅ 완료! ({elapsed:.1f}s)", state="complete", expanded=False)
                    st.success("준비 완료! 이제 질문하세요.")
                    time.sleep(1)
                    st.rerun()
                else:
                    status.update(label="❌ 실패", state="error")
                    st.error("문서를 가져오지 못했습니다.")

# --- 4. 메인 채팅 UI ---
st.title(f"⌨️ Ctrl + F5: {selected_stack}")
st.caption("Enter 키로 지식을 채우고, 최신 문서를 기반으로 코딩하세요.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ .env 파일에 OPENAI_API_KEY를 설정해주세요.")
    st.stop()

# 벡터 DB 로드
vectorstore = load_vectorstore(selected_stack)

# 대화 기록 렌더링
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# 질문 입력
if prompt := st.chat_input(f"{selected_stack} 전문가에게 질문하기..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        # DB가 비어있는지 확인 (Chroma는 컬렉션이 없어도 객체는 생성되므로 get으로 체크 권장되나, 여기선 try-except로 처리)
        history = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages[-10:]
        ]
        
        with st.spinner("최신 문서 분석 중..."):
            try:
                rag_chain = get_rag_chain(vectorstore, selected_stack)
                # 실제 검색 수행
                response = rag_chain.invoke({"input": prompt, "chat_history": history})
                
                # 답변 출력
                st.markdown(response['answer'])
                
                # 근거(Source) 시각화
                with st.expander("🔍 참조한 문서 확인 (Source)"):
                    if response.get('context'):
                        for doc in response['context']:
                            source = doc.metadata.get('source', 'Unknown')
                            st.markdown(f"**🔗 Source:** `{source}`")
                            st.caption(doc.page_content[:250].replace("\n", " ") + "...")
                            st.divider()
                    else:
                        st.write("문서에서 직접적인 정보를 찾지 못했습니다.")
                
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                
            except Exception as e:
                st.warning("⚠️ 아직 학습된 문서가 없거나 검색에 실패했습니다.")
                st.error(f"Error Details: {e}")