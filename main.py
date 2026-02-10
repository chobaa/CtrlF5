import streamlit as st
import os
from dotenv import load_dotenv

# 내부 모듈
from core.database import load_vectorstore, load_config
from core.engine import get_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

# UI & Logic 모듈
from ui.sidebar import render_sidebar
from ui.chat import render_chat_messages, render_input_area

# 1. 설정 및 스타일 로드
load_dotenv()
st.set_page_config(page_title="Ctrl + F5: DevRAG", page_icon="⌨️", layout="wide")

def load_css(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style.css")

# 2. 상태 초기화
if "stacks" not in st.session_state:
    st.session_state.stacks = load_config()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. 사이드바 렌더링
selected_stack, strict_mode, top_k = render_sidebar()

# 4. 메인 영역 헤더
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

# 5. RAG 및 채팅 로직
vectorstore = load_vectorstore(selected_stack)

# 기존 메시지 출력
render_chat_messages(st.session_state.messages)

# RAG 답변 생성
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_msg = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
        with st.spinner("최신 문서 분석 중..."):
            try:
                rag_chain = get_rag_chain(vectorstore, selected_stack, is_strict=strict_mode, relevance_threshold=0.5, top_k=top_k)
                response = rag_chain.invoke({"input": last_user_msg, "chat_history": history})
                answer_text = response['answer']
                
                if strict_mode and "제공된 문서에서 해당 내용에 대한 근거를 찾을 수 없습니다" in answer_text:
                    st.warning("⚠️ **정보 부족 (Strict Mode)**\n\n문서에서 근거를 찾지 못했습니다.")
                    st.session_state.messages.append({"role": "assistant", "content": answer_text})
                else:
                    st.markdown(answer_text)
                    
                    # 참조 문서 데이터 추출 및 저장
                    sources_data = []
                    if response.get('context'):
                        with st.expander("🔍 참조 문서 (Source)"):
                            for i, doc in enumerate(response['context']):
                                score = doc.metadata.get('relevance_score', 0.0)
                                source = doc.metadata.get('source', '알 수 없음')
                                content = doc.page_content
                                
                                # 데이터 저장용
                                sources_data.append({
                                    "source": source,
                                    "content": content,
                                    "score": score
                                })
                                
                                # 즉시 렌더링
                                st.markdown(f"**🔗 출처 {i+1}:** `{source}` (Score: {score:.4f})")
                                st.caption(content[:250].replace("\n", " ") + "...")
                                st.divider()
                    elif not strict_mode:
                        st.info("일반 지식 활용")

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer_text,
                        "sources": sources_data
                    })

            except Exception as e:
                st.error(f"오류 발생: {e}")

# 6. 입력 영역
render_input_area()