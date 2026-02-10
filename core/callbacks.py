import streamlit as st
from core.database import save_config

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
