import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from langchain_redis import RedisSemanticCache
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# ENV SETUP
# -----------------------------
load_dotenv()

st.set_page_config(
    page_title="Semantic QA Reranker",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    llm = ChatOllama(model="llama3.2")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Optional Redis Cache
    try:
        set_llm_cache(
            RedisSemanticCache(
                redis_url="redis://localhost:6379",
                embeddings=embedding_model,
                distance_threshold=0.05,
                ttl=120
            )
        )
    except Exception as e:
        print("Redis not connected:", e)

    return llm, embedding_model, reranker


# -----------------------------
# RETRIEVAL PIPELINE
# -----------------------------
@st.cache_resource(show_spinner="Searching knowledge base...")
def build_retriever(question, _embedding_model):
    loader = WikipediaLoader(query=question + " wikipedia", load_max_docs=10)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
        split_docs,
        _embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 20}
    )

    return retriever


# -----------------------------
# UI
# -----------------------------
st.title("🧠 Semantic Question Answering with Reranker")
st.markdown("""
Ask any question directly.  
This app uses:

✅ Wikipedia Retrieval  
✅ FAISS Vector Search  
✅ CrossEncoder Reranking  
✅ Llama 3.2 Local LLM  
✅ Redis Semantic Cache  
""")

llm, embedding_model, reranker = load_models()

query = st.text_input(
    "Ask Any Question:",
    placeholder="Example: Which animal is the national animal of India?"
)

if st.button("Search & Answer", type="primary") and query:

    try:
        with st.spinner("Retrieving relevant documents..."):
            retriever = build_retriever(query, embedding_model)
            retrieved_docs = retriever.invoke(query)

        with st.spinner("Reranking top results..."):
            pairs = [(query, doc.page_content) for doc in retrieved_docs]
            scores = reranker.predict(pairs)

            ranked = sorted(
                zip(retrieved_docs, scores),
                key=lambda x: x[1],
                reverse=True
            )

            top_docs = [doc for doc, score in ranked[:10]]

        context = "\n\n".join(
            [doc.page_content for doc in top_docs]
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are a precise knowledge assistant.

Answer ONLY from the given context.

If answer not found, say:
I cannot find the answer in the provided context.

Context:
{context}
"""
            ),
            ("user", "{question}")
        ])

        with st.spinner("Generating final answer..."):
            chain = prompt | llm
            response = chain.invoke({
                "context": context,
                "question": query
            })

        # OUTPUT
        st.subheader("💡 Final Answer")
        st.success(response.content)

        st.markdown("---")

        with st.expander("🔍 View Top Retrieved Chunks"):
            for i, doc in enumerate(top_docs, start=1):
                st.markdown(f"### Chunk {i}")
                st.write(doc.page_content)
                st.divider()

    except Exception as e:
        st.error(f"Error: {e}")
