import streamlit as st
import requests
import json
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

ACCENT_COLOR = "#FF6B35"
SIDEBAR_BG = "#2C2C2C"
SIDEBAR_TEXT = "#FFFFFF"
BACKGROUND_COLOR = "#1A1A1A"
MAIN_TEXT_COLOR = "#E8E8E8"
USER_MSG_BG = "#2A4A5E"
ASSISTANT_MSG_BG = "#252525"
INPUT_BG = "#2F2F2F"
BUTTON_COLOR = "#FF6B35"
BUTTON_HOVER = "#FF8C5A"

st.set_page_config(
    page_title="Medical AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

custom_css = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        color: {MAIN_TEXT_COLOR};
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {MAIN_TEXT_COLOR};
        font-weight: 600;
    }}
    
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {SIDEBAR_BG} 0%, #1F1F1F 100%);
        border-right: 1px solid #333;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.3);
    }}
    
    [data-testid="stSidebar"] * {{
        color: {SIDEBAR_TEXT};
    }}
    
    [data-testid="stSidebar"] h3 {{
        color: {ACCENT_COLOR};
        font-weight: 600;
        margin-bottom: 1rem;
    }}
    
    [data-testid="stSidebar"] hr {{
        border-color: #404040;
        margin: 1rem 0;
    }}
    
    .stButton>button {{
        background: linear-gradient(135deg, {BUTTON_COLOR} 0%, {BUTTON_HOVER} 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(255, 107, 53, 0.2);
        width: 100%;
    }}
    
    .stButton>button:hover {{
        background: linear-gradient(135deg, {BUTTON_HOVER} 0%, {BUTTON_COLOR} 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 107, 53, 0.3);
    }}
    
    [data-testid="stChatInput"] {{
        background-color: {INPUT_BG};
        border-radius: 12px;
        border: 1px solid #404040;
        padding: 8px;
    }}
    
    [data-testid="stChatInput"] textarea {{
        background-color: transparent;
        color: {MAIN_TEXT_COLOR} !important;
    }}
    
    [data-testid="stChatMessage"] {{
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        animation: fadeIn 0.3s ease-in;
    }}
    
    [data-testid="stChatMessage"] p {{
        color: {MAIN_TEXT_COLOR} !important;
        font-size: 1rem;
        line-height: 1.6;
    }}
    
    [data-testid="stChatMessage"] * {{
        color: {MAIN_TEXT_COLOR} !important;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .stChatMessage {{
        background-color: {ASSISTANT_MSG_BG};
        border-left: 3px solid #555;
        margin-right: 15%;
    }}
    
    .stChatMessage[data-testid*="user"] {{
        background: linear-gradient(135deg, {USER_MSG_BG} 0%, #234052 100%);
        margin-left: 15%;
        margin-right: 0;
        border-left: 3px solid {ACCENT_COLOR};
    }}
    
    .highlight {{
        color: {ACCENT_COLOR} !important;
        font-weight: 600;
    }}
    
    .title-container {{
        display: flex;
        align-items: center;
        padding: 1.5rem 0 1rem 0;
        margin-bottom: 0.5rem;
    }}
    
    .logo-text {{
        font-size: 2.8rem;
        font-weight: 700;
        color: {MAIN_TEXT_COLOR};
        letter-spacing: -1px;
    }}
    
    .logo-dot {{
        color: {ACCENT_COLOR};
    }}
    
    .subtitle {{
        color: #B0B0B0;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }}
    
    .caption {{
        color: #808080;
        font-size: 0.85rem;
    }}
    
    .status-badge {{
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }}
    
    .status-connected {{
        background-color: rgba(76, 175, 80, 0.2);
        color: #4CAF50;
        border: 1px solid #4CAF50;
    }}
    
    .status-disconnected {{
        background-color: rgba(244, 67, 54, 0.2);
        color: #F44336;
        border: 1px solid #F44336;
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def get_new_session_id():
    try:
        response = requests.get(f"{BACKEND_URL}/new_session")
        response.raise_for_status()
        return response.json().get("session_id")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to backend: {e}")
        return None

def send_message_to_backend(session_id: str, message: str):
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"session_id": session_id, "message": message}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with backend: {e}")
        return None

if "session_id" not in st.session_state:
    st.session_state.session_id = get_new_session_id()
    if st.session_state.session_id:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your post-discharge medical assistant. To start, please tell me your <span class='highlight'>full name</span> or <span class='highlight'>patient ID</span>."}
        ]
    else:
        st.session_state.messages = [
            {"role": "assistant", "content": "Error: Could not start session. Please ensure the backend is running."}
        ]

st.markdown(
    """
    <div class="title-container">
        <span class="logo-text">Medi<span class="logo-dot">AI</span></span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="subtitle">Post-Discharge Medical Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="caption">Powered by LangGraph, FastAPI, Streamlit, and Gemini 2.5 Flash</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and ("full name" in message["content"] or "patient ID" in message["content"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your discharge instructions or condition...", disabled=not st.session_state.session_id):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.session_id:
        with st.spinner("Assistant is thinking..."):
            backend_response = send_message_to_backend(st.session_state.session_id, prompt)

        if backend_response:
            assistant_response = backend_response.get("response")
            
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    else:
        with st.chat_message("assistant"):
            st.markdown("Cannot send message. Backend is not connected.")
    
    st.rerun()

with st.sidebar:
    st.markdown("<h3>Session Control</h3>", unsafe_allow_html=True)
    
    status_html = f"<span class='status-badge status-connected'>Connected</span>" if st.session_state.session_id else "<span class='status-badge status-disconnected'>Disconnected</span>"
    st.markdown(status_html, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    if st.button("Start New Chat"):
        st.session_state.session_id = get_new_session_id()
        if st.session_state.session_id:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I am your post-discharge medical assistant. To start, please tell me your <span class='highlight'>full name</span> or <span class='highlight'>patient ID</span>."}
            ]
        else:
            st.session_state.messages = [
                {"role": "assistant", "content": "Error: Could not start session. Please ensure the backend is running."}
            ]
        st.rerun()