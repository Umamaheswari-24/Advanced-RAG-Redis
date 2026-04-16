import os
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from sentence_transformers import CrossEncoder
from langchain_redis import RedisSemanticCache
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import ChatPromptTemplate

os.environ["USER_AGENT"] = "MyRerankerApp/1.0"
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

st.set_page_config(page_title="Wikipedia QA Reranker", page_icon="📚")

# Cache models and data so they aren't painfully reloaded on every keystroke
@st.cache_resource(show_spinner="Starting neural engines and caching data...")
def init_models_and_data():
    llm = ChatOllama(model="tinyllama")
    
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Paracetamol") 
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    try:
        set_llm_cache(RedisSemanticCache(
            redis_url="redis://localhost:6379",
            embeddings=embedding_model,
            distance_threshold=0.05,
            ttl=120
        ))
    except Exception as e:
        print(f"Warning: Could not connect to Redis cache: {e}")
        
    return llm, retriever, reranker

st.title("📚 Wikipedia AI Reranker")
st.markdown("Ask anything about **Paracetamol** based on its Wikipedia article. This app uses FAISS semantic search, a neural CrossEncoder for reranking relevance, and TinyLlama for local summarization.")

# This block will take ~10-20 seconds the very first time the app spins up, then load instantly.
llm, retriever, reranker = init_models_and_data()

# User Input
query = st.text_input("What would you like to know?", placeholder="e.g. What are the key side effects mentioned?")

if st.button("Search & Answer", type="primary") and query:
    st.divider()
    
    with st.spinner("Retrieving initial closest 20 documents via vector embeddings..."):
        retrieved_docs = retriever.invoke(query)
    
    with st.spinner("Neural Reranking top 5 chunks with ms-marco CrossEncoder..."):
        pairs = [(query, doc.page_content) for doc in retrieved_docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in ranked[:5]]
    
    context = "\n\n".join([doc.page_content for doc in top_docs])
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a highly precise medical assistant. Your ONLY job is to answer the user's question based strictly on the provided Context. If the context does not contain the answer, say 'I cannot find the answer in the provided context.' Do not make up information.\n\nContext:\n{context}"),
        ("user", "{question}")
    ])
    
    with st.spinner("Drafting answer usign Local TinyLlama..."):
        chain = prompt_template | llm
        response = chain.invoke({"context": context, "question": query})
    
    # Results
    st.subheader("💡 Answer")
    st.info(response.content)
    
    # Expandable Context blocks
    st.markdown("---")
    with st.expander("🔍 View Retrieved Context Chunks"):
        st.write("These are the most highly-rated chunks curated by the CrossEncoder for the LLM.")
        for i, doc in enumerate(top_docs):
            st.markdown(f"**Chunk {i+1}**")
            st.caption(doc.page_content)
            st.divider()