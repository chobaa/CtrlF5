# assets/styles.py

MODERN_CSS = """
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
"""