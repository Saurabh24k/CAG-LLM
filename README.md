# 💬 Cache Augmented Generation (CAG) Chatbot

**An intelligent, professional, and visually intuitive Chatbot using Cache Augmented Generation (CAG) for faster and smarter LLM responses.**  
This project demonstrates how to enhance language model efficiency using caching, embeddings, and real-time performance monitoring.

---

## 📖 Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Cache Mechanism](#cache-mechanism)
- [How It Works](#how-it-works)
- [Future Enhancements](#future-enhancements)

---

## 📦 Project Overview

The **Cache Augmented Generation (CAG)** Chatbot is a professional chatbot designed to reduce response time and improve performance by using **smart caching mechanisms** for language model responses. It showcases:

- Efficient data caching with embeddings.
- Real-time performance monitoring.
- Optimized for **LLM inference** and **reduced latency**.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Backend:** Python with subprocess-based LLM querying
- **LLM Integration:** Demo : Mistral-7B-Instruct-v0.3, Offline : LLaMA3 and Ollama (configurable in `generation_model.py`)
- **Data Handling:** NumPy, Pandas
- **Visualization:** Plotly, Streamlit Components
- **Embedding Generation:** Custom vector embedding methods
- **Version Control:** Git, GitHub

---

## 📐 Architecture

```plaintext
📦 cag-demo
├── 📂 src
│   ├── cache_manager.py       # Cache management logic (Singleton Pattern)
│   ├── generation_model.py    # Core model handling and cache interaction
│   ├── embedding_utils.py     # Embedding generation and similarity calculation
│   └── app.py                 # Streamlit application and UI logic
├── requirements.txt           # Python dependencies
├── .streamlit/config.toml     # Custom Streamlit theme configuration
├── README.md                  # Project documentation
└── 📦 tests                   # Unit tests (optional, recommended for production)
```

---

## 📥 Installation

To run the CAG Chatbot locally, follow these steps:

### **Prerequisites:**
- Python 3.10+
- Streamlit
- Git
- Ollama

### **Steps:**
```bash
# Clone the repository
git clone https://github.com/yourusername/cag-chatbot.git
cd cag-chatbot

# Create a virtual environment
python -m venv cag-env
source cag-env/bin/activate  # For Mac/Linux
# .\cag-env\Scripts\activate  # For Windows

# Install dependencies
pip install -r requirements.txt

# Install Ollama
pip install ollama
```

---

## 🚀 Usage

### **Run the Chatbot Locally:**
```bash
streamlit run src/app.py
```

**Interacting with the Chatbot:**
- Enter your query in the main chat panel.
- Monitor cache performance and statistics on the side panel.
- Adjust cache size and similarity threshold using the configurator.

---

## 🧠 Cache Mechanism

The caching system uses a singleton cache manager with the following steps:

1. **Exact Match:** If a query matches an existing cached key, it returns the cached response.
2. **Embedding Similarity:** If a query is semantically similar (above a configurable threshold), the cached response is returned.
3. **Cache Miss:** If no match is found, the LLM is queried, and the result is cached.

**Cache Eviction Strategy:**
- Least Recently Used (LRU) eviction occurs when the cache capacity exceeds the limit.

---

## 🛡️ How It Works (Step-by-Step)

1. **Input Query:** The user inputs a query in the chatbot.
2. **Cache Check:** The system checks the cache for an exact match.
3. **Embedding Generation:** If no match, an embedding is generated for similarity checking.
4. **LLM Query:** If no approximate match is found, the system queries the language model.
5. **Caching the Response:** The response is cached along with the generated embedding.
6. **Monitoring:** Real-time performance metrics and visualizations are updated in the UI.

---

## 🚧 Future Enhancements

- 🔧 Integration with more LLMs like GPT-4, PaLM, and Claude.
- 🔧 Implement a distributed caching system for scalability.
- 🔧 Add support for additional languages and models.

---


