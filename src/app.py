import sys
import os
import time
import streamlit as st
import plotly.express as px
from datetime import datetime
from src.generation_model import LLMIntegration
from dotenv import load_dotenv
import os

load_dotenv()

# Set environment variables to prevent deadlocks and parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Initialize LLM Integration
llm_system = LLMIntegration()

# Cache statistics and tracking
if "cache_hits" not in st.session_state:
    st.session_state.cache_hits = 0
    st.session_state.cache_misses = 0
    st.session_state.response_times = []
    st.session_state.query_timestamps = []
    st.session_state.history = []

# Streamlit Page Configuration
st.set_page_config(page_title="CAG Chatbot", layout="wide")
st.title("💬 Cache Augmented Generation Chatbot")
st.write("An intelligent chatbot with cache-enhanced LLM responses.")

# Layout Columns
col1, col2, col3 = st.columns([1.2, 2, 1.2])

# Configurator Section (Right Pane)
with col1:
    st.header("⚙️ Configurator")
    cache_size = st.slider("Cache Size", min_value=50, max_value=500, value=100)
    similarity_threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=1.0, value=0.8)
    clear_cache = st.button("🗑️ Clear Cache")
    
    if clear_cache:
        llm_system.cache_manager.clear_cache()
        st.session_state.cache_hits = 0
        st.session_state.cache_misses = 0
        st.session_state.response_times = []
        st.session_state.query_timestamps = []
        st.session_state.history = []
        st.success("Cache cleared!")

    # Collapsible Cache Content Section
    with st.expander("📦 View Cache Content"):
        if llm_system.cache_manager.cache:
            for key, value in llm_system.cache_manager.cache.items():
                st.write(f"**Query:** {key}")
                st.write(f"**Response:** {value['response']}")
                st.write(f"**Timestamp:** {datetime.fromtimestamp(value['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                st.write("---")
        else:
            st.write("Cache is empty.")

# Main Chat Section (Middle Pane)
with col2:
    st.header("💡 Chatbot Interaction")
    query = st.text_input("Enter your query:")
    if query:
        with st.spinner("Processing... Fetching from Cache or LLM..."):
            start_time = time.time()
            response = llm_system.generate_response(query)
            response_time = time.time() - start_time

            # Determine cache hit or miss
            if "Cache Hit!" in response:
                st.session_state.cache_hits += 1
            else:
                st.session_state.cache_misses += 1

            # Log query details
            st.session_state.response_times.append(response_time)
            st.session_state.query_timestamps.append(datetime.now().strftime('%H:%M:%S'))
            st.session_state.history.append({"query": query, "response": response, "time": response_time})

            # Display response in chat-style
            st.success(response)
            st.info(f"⏱️ Response Time: {response_time:.2f} seconds")

    # Query History Section (Collapsible)
    with st.expander("🕰️ Query History"):
        for entry in st.session_state.history[-10:]:
            st.write(f"**Query:** {entry['query']}")
            st.write(f"**Response:** {entry['response']}")
            st.write(f"⏱️ Time Taken: {entry['time']:.2f} seconds")

# Cache Statistics Section (Left Pane)
with col3:
    st.header("📊 Cache Statistics")

    # Metrics Display
    col1_stat, col2_stat, col3_stat = st.columns(3)
    col1_stat.metric("✅ Cache Hits", st.session_state.cache_hits)
    col2_stat.metric("❌ Cache Misses", st.session_state.cache_misses)
    col3_stat.metric("📦 Cache Size", len(llm_system.cache_manager.cache))

    # Cache Hit/Miss Ratio
    total_queries = st.session_state.cache_hits + st.session_state.cache_misses
    hit_ratio = (st.session_state.cache_hits / total_queries) * 100 if total_queries > 0 else 0
    miss_ratio = (st.session_state.cache_misses / total_queries) * 100 if total_queries > 0 else 0
    st.progress(hit_ratio / 100, text=f"Cache Hit Ratio: {hit_ratio:.2f}%")
    st.progress(miss_ratio / 100, text=f"Cache Miss Ratio: {miss_ratio:.2f}%")

    # Response Time Graph
    if st.session_state.response_times:
        fig = px.line(x=st.session_state.query_timestamps, y=st.session_state.response_times,
                      title="Response Time Over Queries",
                      labels={"x": "Timestamp", "y": "Response Time (s)"})
        st.plotly_chart(fig)

# Footer Styling
st.markdown("---")
st.write("Created with ❤️ by Saurabh Rajput for an elegant and professional demonstration of Cache Augmented Generation.")
