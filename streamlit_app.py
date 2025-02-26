import os
import streamlit as st
import shutil
from datetime import datetime
from app import ContentEngine

st.set_page_config(page_title="Document Analysis Chat", layout="wide")

# Ensure pdf directory exists
pdf_dir = "./pdfs"
os.makedirs(pdf_dir, exist_ok=True)

# Function to upload PDFs and save them to the 'pdfs' directory
def upload_pdfs():
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(pdf_dir, uploaded_file.name)
            # Save file to pdfs directory
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Uploaded and saved successfully!")

# Render the uploader above the chat interface
st.sidebar.header("Upload PDFs")
upload_pdfs()

st.markdown(
    """
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        width: 100%;
    }
    .user-message-container {
        display: flex;
        justify-content: flex-end;
        margin: 5px 0;
    }
    .assistant-message-container {
        display: flex;
        justify-content: flex-start;
        margin: 5px 0;
    }
    .user-message {
        background-color: #dcf8c6;
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        margin-right: 15px;
        color: #000;
    }
    .assistant-message {
        background-color: #fff;
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        margin-left: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        color: #000;
    }
    .message-time {
        color: #999;
        font-size: 0.8em;
        margin-top: 5px;
        text-align: right;
    }
    .source-citation {
        font-size: 0.8em;
        color: #666;
        border-left: 3px solid #ccc;
        padding-left: 10px;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 Document Analysis Chat")

@st.cache_resource
def get_content_engine():
    return ContentEngine()

engine = get_content_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

def render_message(role, content, timestamp, sources=""):
    if role == "user":
        st.markdown(f"""
        <div class="chat-container">
            <div class="user-message-container">
                <div class="user-message">
                    {content}
                    <div class="message-time">{timestamp}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-container">
            <div class="assistant-message-container">
                <div class="assistant-message">
                    {content}
                    <div class="message-time">{timestamp}</div>
                    {f'<div class="source-citation">{sources}</div>' if sources else ''}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Render previous chat messages
for message in st.session_state.messages:
    render_message(message["role"], message["content"], message["timestamp"], message.get("sources", ""))

if prompt := st.chat_input("Type your question here..."):
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": timestamp
    })
    render_message("user", prompt, timestamp)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = engine.query(prompt)
            if isinstance(result, dict):
                answer = result.get('answer', 'No answer provided')
                sources = "\n".join([
                    f"Source {i+1}: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})" 
                    for i, doc in enumerate(result.get('source_documents', []))
                ])
            else:
                answer = result
                sources = ""
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": timestamp
            })
            render_message("assistant", answer, timestamp, sources)