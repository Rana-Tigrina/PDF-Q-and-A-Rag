import streamlit as st
from app import ContentEngine
from datetime import datetime

st.set_page_config(page_title="Document Analysis Chat", layout="wide")

# Custom CSS for WhatsApp-like styling
st.markdown("""
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
    color: #000000;  /* Black text color */
}
.assistant-message {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 10px;
    max-width: 70%;
    margin-left: 15px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    color: #000000;  /* Black text color */
}
.message-time {
    color: #999999;
    font-size: 0.8em;
    float: right;
    margin-top: 5px;
}
.source-citation {
    font-size: 0.8em;
    color: #666666;
    border-left: 3px solid #ccc;
    padding-left: 10px;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 Document Analysis Chat")

# Initialize the content engine
@st.cache_resource
def get_content_engine():
    return ContentEngine()

engine = get_content_engine()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-container">
            <div class="user-message-container">
                <div class="user-message">
                    {message["content"]}
                    <div class="message-time">{message["timestamp"]}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-container">
            <div class="assistant-message-container">
                <div class="assistant-message">
                    {message["content"]}
                    <div class="message-time">{message["timestamp"]}</div>
                    <div class="source-citation">{message.get('sources', '')}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Add user message
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    with st.chat_message("user"):
        st.markdown(f"""
        <div class="chat-container">
            <div class="user-message-container">
                <div class="user-message">
                    {prompt}
                    <div class="message-time">{datetime.now().strftime("%H:%M")}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = engine.query(prompt)
            if isinstance(result, dict):
                answer = result['answer']
                sources = "\n".join([
                    f"Source {i+1}: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})" 
                    for i, doc in enumerate(result.get('source_documents', []))
                ])
            else:
                answer = result
                sources = ""

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": datetime.now().strftime("%H:%M")
            })

            st.markdown(f"""
            <div class="chat-container">
                <div class="assistant-message-container">
                    <div class="assistant-message">
                        {answer}
                        <div class="message-time">{datetime.now().strftime("%H:%M")}</div>
                        <div class="source-citation">{sources}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)