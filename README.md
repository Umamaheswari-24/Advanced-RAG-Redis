# Advanced RAG with Reranker & TTL Cache using Redis

## 📌 Project Overview
This project is an **Advanced Retrieval-Augmented Generation (RAG)** application built using **Python, LangChain, Redis, FAISS, HuggingFace, and Streamlit**. It retrieves relevant information from a knowledge source, reranks results using a CrossEncoder model, and generates accurate answers using a local LLM (**TinyLlama**). The project also uses **Redis Semantic Cache with TTL (Time-To-Live)** to improve speed and reduce repeated LLM calls.

---

## 🚀 Features
- Semantic document retrieval using **FAISS**
- Intelligent reranking using **CrossEncoder**
- Fast response with **Redis Semantic Cache**
- TTL support for automatic cache expiration
- Local LLM answer generation using **TinyLlama**
- Streamlit web interface
- Wikipedia-based document source

---

## 🛠️ Tech Stack
- Python
- Streamlit
- LangChain
- Redis
- FAISS
- HuggingFace Embeddings
- Sentence Transformers
- TinyLlama (Ollama)

---

## 📂 Project Structure
Advanced-RAG-Redis/  
│── Reranker_web.py  
│── requirements.txt  
│── README.md  

---

## ⚙️ Installation & Setup

1. Clone Repository
git clone https://github.com/yourusername/Advanced-RAG-Redis.git
cd Advanced-RAG-Redis

2. Install Dependencies
pip install -r requirements.txt

3. Start Redis
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

4. Install Ollama & Pull TinyLlama
ollama pull tinyllama

5. Run Application
streamlit run Reranker_web.py

💡 How It Works
User enters a query
FAISS retrieves top relevant chunks
CrossEncoder reranks retrieved chunks
TinyLlama generates final answer
Redis caches responses with TTL


📸 Sample Use Cases

<img width="1261" height="575" alt="Screenshot 2026-04-16 221951" src="https://github.com/user-attachments/assets/69cf2a40-ac2d-48a0-8d62-fb4e6c989b85" />
<img width="1257" height="549" alt="Screenshot 2026-04-16 222012" src="https://github.com/user-attachments/assets/503959e9-a536-4d26-b8b1-503946cbe891" />
<img width="1221" height="568" alt="Screenshot 2026-04-16 222044" src="https://github.com/user-attachments/assets/0a5036e7-0301-4f55-a9c5-12e250346508" />
<img width="1229" height="242" alt="Screenshot 2026-04-16 222103" src="https://github.com/user-attachments/assets/ed8a894f-334c-428f-8a42-d463618d18c1" />
<img width="1250" height="568" alt="Screenshot 2026-04-16 222324" src="https://github.com/user-attachments/assets/37b500ed-425b-4ce1-9c5f-ef794bf568ff" />
