# 🛍 DeepShiva - Domain-Specific Tourism Chatbot

A multilingual (English + Tamil) AI chatbot that answers tourism-related queries about Tamil Nadu using a custom Q\&A dataset, semantic search with FAISS, and a beautiful Streamlit frontend.

---

## 🚀 Features

* 🔍 Intelligent tourism Q\&A using semantic search (FAISS + Sentence Transformers)
* 🌐 Supports English & Tamil queries automatically
* 🧠 Merged logic (backend + semantic search) directly in Streamlit app
* 💬 Frontend built with Streamlit
* 🖼️ Custom background image and chat UI
* 🗃️ Local embedding index for blazing-fast performance

---

## 📁 Project Structure

```
TOURISM-CHATBOT/
├── app.py                      # Merged Streamlit app (frontend + backend)
├── Thanjai.jpg                 # Background image
├── full_tamil_nadu_tourism_qa.json   # English QA dataset
├── tamil_tourism_qa_full.json        # Tamil QA dataset
├── tourism_all_index.faiss     # FAISS index
├── tourism_all_data.pkl        # Pickled embedding data
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Installation & Run

### 1. Clone the repo

```bash
git clone https://github.com/Vasundhara2508/Chatbot.git
cd Chatbot
```

### 2. (Optional) Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # On Windows
source venv/bin/activate     # On Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the chatbot

```bash
streamlit run app.py
```

> ✅ This will start the chatbot UI in your browser.

---

## 🌍 Language Detection

The app auto-detects Tamil characters (Unicode range) and routes to the correct dataset:

| Input Language | Dataset Used                        |
| -------------- | ----------------------------------- |
| English        | full\_tamil\_nadu\_tourism\_qa.json |
| Tamil          | tamil\_tourism\_qa\_full.json       |

---

## 🧲 Example Questions

* **English:**
  `Tell me about the Chola architecture in Tamil Nadu.`

* **Tamil:**
  `காஞ்சிபுரத்தில் உள்ள கோயில்கள் பத்தி கூறவும்`

---

## 🧹 .gitignore

```gitignore
__pycache__/
*.pkl
*.faiss
venv/
.env
```
