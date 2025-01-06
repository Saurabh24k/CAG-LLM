import sys
import os
import time
import streamlit as st
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv

# Import the LLM Integration Model
from generation_model import LLMIntegration

# Load environment variables and secrets
load_dotenv()
api_key = st.secrets.get("general", {}).get("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY"))

# Validate API Key at Initialization
if not api_key:
    st.error("❌ API key is missing. Please add it to the secrets manager or your .env file.")
    st.stop()

# Initialize LLM Integration with API Key
llm_system = LLMIntegration(api_key=api_key)

# Cache statistics and tracking initialization
if "cache_hits" not in st.session_state:
    st.session_state.cache_hits = 0
    st.session_state.cache_misses = 0
    st.session_state.response_times = []
    st.session_state.query_timestamps = []
    st.session_state.history = []

st.set_page_config(
    page_title="CAG Chatbot", 
    layout="wide", 
    page_icon="🧀", 
    initial_sidebar_state="expanded"
)

# CSS for Styling Graph
st.markdown(
    """
    <style>
        body { font-family: 'Arial', sans-serif; }
        .stTextInput, .stButton { border-radius: 8px; }
        .stProgress > div > div { border-radius: 20px; }
        .custom-link { color: #1f77b4; text-decoration: none; font-weight: bold; transition: color 0.3s ease-in-out; }
        .custom-link:hover { color: #ff4b4b; }
        .fixed-graph-container { max-height: 300px !important; overflow-y: auto; }
    </style>
    """,
    unsafe_allow_html=True
)

# Page Title and Description
st.title("💡 Cache Augmented Generation (CAG) Chatbot")
st.write("**A chatbot with enhanced responses powered by smart caching.**")

# Layout Columns: Configurator | Chat | Statistics
col1, col2, col3 = st.columns([1.2, 2, 1.2])

# 🛠️ **Configurator Section (Left Panel)**
with col1:
    st.header("⚙️ Configurator")
    cache_size = st.slider("🗄️ Cache Size", min_value=50, max_value=500, value=100)
    similarity_threshold = st.slider("📈 Similarity Threshold", min_value=0.5, max_value=1.0, value=0.8)
    clear_cache = st.button("🧹 Clear Cache")

    if clear_cache:
        llm_system.cache_manager.clear_cache()
        st.session_state.cache_hits = 0
        st.session_state.cache_misses = 0
        st.session_state.response_times = []
        st.session_state.query_timestamps = []
        st.session_state.history = []
        st.success("✅ Cache cleared successfully!")

    # 📦 **Cache Content Section**
    with st.expander("📦 **View Cache Content**"):
        if llm_system.cache_manager.cache:
            for key, value in llm_system.cache_manager.cache.items():
                st.write(f"**Query:** {key}")
                st.write(f"**Response:** {value['response']}")
                st.write(f"**Timestamp:** {datetime.fromtimestamp(value['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                st.write("---")
        else:
            st.write("🗑️ Cache is currently empty.")

# 💬 **Chat Interaction Section (Middle Panel)**
with col2:
    st.header("💬 Chat with CAG")
    query = st.text_input("💡 Enter your query:")
    if query:
        start_time = time.time()

        # Step 1: Check Cache
        st.info("⏳ Checking Cache...")
        cached_response = llm_system.cache_manager.get_from_cache(llm_system.cache_manager.normalize_key(query))
        
        if cached_response:
            # Step 2: If Cache Hit, Return
            st.success("✅ Cache Hit! Returning cached response.")
            response = cached_response
            st.session_state.cache_hits += 1
        else:
            # Step 3: If Cache Miss, Query LLM
            st.warning("❌ Cache Miss. Fetching from LLM...")
            response = llm_system.generate_response(query)
            st.session_state.cache_misses += 1

        # Response Time and Save Data
        response_time = time.time() - start_time
        st.session_state.response_times.append(response_time)
        st.session_state.query_timestamps.append(datetime.now().strftime('%H:%M:%S'))
        st.session_state.history.append({"query": query, "response": response, "time": response_time})

        # 🎯 Chat Response
        st.success(f"**🗨️ {response}**")
        st.info(f"⏱️ **Response Time:** {response_time:.2f} seconds")

    # 📜 **Query History Section**
    with st.expander("🕰️ **Query History**"):
        for entry in st.session_state.history[-10:]:
            st.write(f"**Query:** {entry['query']}")
            st.write(f"**Response:** {entry['response']}")
            st.write(f"⏱️ **Time Taken:** {entry['time']:.2f} seconds")
            st.write("---")

# 📊 **Cache Statistics Section (Right Panel)**
with col3:
    st.header("📊 Cache Statistics")

    # Real-Time Metrics
    col1_stat, col2_stat, col3_stat = st.columns(3)
    col1_stat.metric("✅ Hits", st.session_state.cache_hits)
    col2_stat.metric("❌ Misses", st.session_state.cache_misses)
    col3_stat.metric("📦 Cache Size", len(llm_system.cache_manager.cache))

    # Cache Hit/Miss Ratio
    total_queries = st.session_state.cache_hits + st.session_state.cache_misses
    hit_ratio = (st.session_state.cache_hits / total_queries) * 100 if total_queries > 0 else 0
    miss_ratio = (st.session_state.cache_misses / total_queries) * 100 if total_queries > 0 else 0

    st.progress(hit_ratio / 100, text=f"✅ Cache Hit Ratio: {hit_ratio:.2f}%")
    st.progress(miss_ratio / 100, text=f"❌ Cache Miss Ratio: {miss_ratio:.2f}%")

    # 📈 **Response Time Graph**
    if st.session_state.response_times:
        st.markdown('<div class="fixed-graph-container">', unsafe_allow_html=True)
        fig = px.line(
            x=st.session_state.query_timestamps,
            y=st.session_state.response_times,
            title="📈 Response Time Trend",
            labels={"x": "Timestamp", "y": "Response Time (s)"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ✅ **Footer**
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p><strong>🚀 Built by 
            <a href="https://www.linkedin.com/in/saurabh-rajput-24k/" 
               target="_blank" 
               class="custom-link">
                Saurabh Rajput
            </a>
            for the demonstration of Cache Augmented Generation.
        </strong></p>
    </div>
    """, 
    unsafe_allow_html=True
)
