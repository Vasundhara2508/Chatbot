import streamlit as st
import os, pickle, faiss
import re, base64
from sentence_transformers import SentenceTransformer

# ------------------------------ Load Model + Index ------------------------------
@st.cache_resource
def load_model_index():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    index = faiss.read_index("tourism_all_index.faiss")
    with open("tourism_all_data.pkl", "rb") as f:
        data = pickle.load(f)
    return model, index, data

model, index, data = load_model_index()

# ------------------------------ Language Detection ------------------------------
def detect_language(text):
    return "ta" if re.search(r'[\u0B80-\u0BFF]', text) else "en"

# ------------------------------ Answer Retrieval ------------------------------
def get_answer_from_query(query):
    try:
        embedding = model.encode([query])
        D, I = index.search(embedding, k=1)
        match = data[I[0][0]]
        return match["answer"]
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------------------ UI Styling ------------------------------
st.set_page_config(page_title="🌄 Tamil Nadu Tourism Chatbot", layout="wide")

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

bg_img = get_base64_image("Thanjai.jpg")
if bg_img:
    st.markdown(f"""
        <style>
            .stApp {{
                background-image: linear-gradient(to bottom, rgba(255,255,255,0.2), rgba(255,255,255,0.3)),
                                  url("data:image/jpg;base64,{bg_img}");
                background-size: cover;
                background-position: center;
                font-family: 'Apple Chancery', cursive !important;
                color: #d3d3d3;
            }}
            .fixed-header {{
                position: fixed;
                top: 0; left: 0; right: 0;
                background-color: rgba(255, 255, 255, 0.15);
                backdrop-filter: blur(10px);
                padding: 1rem; text-align: center;
                font-size: 2.2vw; font-weight: bold;
                border-bottom: 1px solid rgba(255,255,255,0.2);
                z-index: 1000;
            }}
            .chat-container {{
                margin-top: 100px; margin-bottom: 120px;
                height: 60vh; overflow-y: auto;
                padding: 0 25px;
            }}
            .user-msg, .bot-msg {{
                padding: 14px 20px; border-radius: 20px;
                max-width: 80%; margin-bottom: 16px;
                font-size: 1.05rem;
            }}
            .user-msg {{ background-color: #DCE775; color: #33691E; margin-left: auto; }}
            .bot-msg {{ background-color: #9575CD; color: white; margin-right: auto; }}
            .chat-input {{
                position: fixed; bottom: 0; left: 0; right: 0;
                background: rgba(255,255,255,0.95); padding: 12px 25px;
                z-index: 999; box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
            }}
            .right-note {{
                position: fixed; top: 110px; right: 20px; width: 300px;
                background-color: rgba(255,255,255,0.15);
                padding: 15px; border-radius: 12px;
                color: black; font-size: 13.5px;
                font-weight: bold; line-height: 1.6;
            }}
        </style>
    """, unsafe_allow_html=True)

# ------------------------------ Header ------------------------------
st.markdown('<div class="fixed-header">🌄 Discover the Wonders of Tamil Nadu – Powered by AI</div>', unsafe_allow_html=True)

# ------------------------------ Chat State ------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------ Sidebar ------------------------------
with st.sidebar:
    st.title("🕘 Chat History")
    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
    for item in reversed(st.session_state.chat_history):
        st.markdown(f"🗨 {item['question'][:30]}...")

# ------------------------------ Right Note ------------------------------
st.markdown("""
    <div class="right-note">
        🤖 Tamil Nadu Tourism AI Assistant<br><br>
        Try asking:<br>
        “Tell me about Meenakshi Temple”<br>
        “பொங்கல் என்பது என்ன?”<br>
        “What’s special in Mahabalipuram?”<br><br>
        Created by: Magna, Vasundhara, Aarmitha, Keerthi
    </div>
""", unsafe_allow_html=True)

# ------------------------------ Chat Window ------------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown('<div class="bot-msg">Hi there! 👋 Ask me about temples, places, or festivals in Tamil Nadu.</div>', unsafe_allow_html=True)

for chat in st.session_state.chat_history:
    st.markdown(f'<div class="user-msg">{chat["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bot-msg">{chat["answer"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ Chat Input ------------------------------
query = st.chat_input("Type your message here...")
if query and query.strip():
    answer = get_answer_from_query(query.strip())
    st.session_state.chat_history.append({"question": query.strip(), "answer": answer})
    st.rerun()
elif query:
    st.warning("Please enter a valid message.")
