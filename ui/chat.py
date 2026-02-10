import streamlit as st
import streamlit.components.v1 as components
from core.callbacks import append_to_prompt, send_message_callback

def render_chat_messages(messages):
    for m in messages:
        with st.chat_message(m["role"]):
            if "제공된 문서에서 해당 내용에 대한 근거를 찾을 수 없습니다" in m["content"]:
                 st.warning(m["content"])
            else:
                 st.markdown(m["content"])
                 
                 # 저장된 참조 문서가 있으면 출력
                 if "sources" in m and m["sources"]:
                     with st.expander("🔍 참조 문서 (Source)"):
                         for i, doc in enumerate(m["sources"]):
                             score = doc.get("score", 0.0)
                             st.markdown(f"**🔗 출처 {i+1}:** `{doc.get('source', '알 수 없음')}` (Score: {score:.4f})")
                             st.caption(doc.get("content", "")[:250].replace("\n", " ") + "...")
                             st.divider()

def render_input_area():
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
    components.html("""
    <script>
        const doc = window.parent.document;
        function clickSendButton() {
            const buttons = Array.from(doc.querySelectorAll('button'));
            const sendButton = buttons.find(el => el.innerText.includes('전송'));
            if (sendButton) sendButton.click();
        }
        doc.addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                e.stopPropagation();
                clickSendButton();
            }
        });
    </script>
    """, height=0, width=0)
