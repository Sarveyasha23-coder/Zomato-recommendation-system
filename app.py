import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Restaurant Discovery", 
    page_icon="🍽️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Enhanced CSS with Modern Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@700;800;900&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        font-family: 'Poppins', sans-serif;
    }
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes glowPulse {
        0%, 100% {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.4);
        }
        50% {
            box-shadow: 0 0 40px rgba(102, 126, 234, 0.8);
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    @keyframes rotate360 {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
    
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1729 100%);
        color: #ffffff;
    }
    
    /* Developer Badge */
    .developer-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 18px 35px;
        border-radius: 50px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.3);
        animation: fadeInDown 0.8s ease-out, glowPulse 3s ease-in-out infinite;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .developer-badge::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }
    
    .developer-badge p {
        color: white;
        font-size: 1.15em;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.8px;
        position: relative;
        z-index: 1;
    }
    
    .developer-badge .emoji {
        font-size: 1.4em;
        margin-right: 10px;
        animation: float 3s ease-in-out infinite;
    }
    
    .developer-badge a {
        color: #ffd700;
        text-decoration: none;
        font-weight: 800;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        margin-left: 10px;
        position: relative;
        z-index: 1;
    }
    
    .developer-badge a::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 0;
        height: 2px;
        background: #ffed4e;
        transition: width 0.4s ease;
    }
    
    .developer-badge a:hover {
        color: #ffed4e;
        text-shadow: 0 0 15px rgba(255, 215, 0, 0.8);
        transform: scale(1.05);
    }
    
    .developer-badge a:hover::after {
        width: 100%;
    }
    
    /* Main Header */
    .main-header {
        text-align: center;
        padding: 50px 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 25px;
        margin-bottom: 35px;
        color: white;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        animation: fadeInUp 0.8s ease-out 0.2s both;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate360 20s linear infinite;
    }
    
    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3.5em;
        margin-bottom: 10px;
        font-weight: 900;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 2;
        letter-spacing: -1px;
    }
    
    .main-header p {
        font-size: 1.25em;
        opacity: 0.95;
        letter-spacing: 1.2px;
        font-weight: 500;
        position: relative;
        z-index: 2;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        backdrop-filter: blur(10px);
        padding: 28px;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.8s ease-out backwards;
    }
    
    .metric-card:nth-child(1) { animation-delay: 0.3s; }
    .metric-card:nth-child(2) { animation-delay: 0.4s; }
    .metric-card:nth-child(3) { animation-delay: 0.5s; }
    .metric-card:nth-child(4) { animation-delay: 0.6s; }
    
    .metric-card:hover {
        transform: translateY(-12px);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    .metric-card .number {
        font-size: 2.8em;
        font-weight: 900;
        margin: 15px 0;
        background: linear-gradient(135deg, #667eea, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card .label {
        font-size: 1em;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    /* Recommendation Card */
    .recommendation-card {
        background: linear-gradient(135deg, rgba(245, 247, 250, 0.08) 0%, rgba(195, 207, 226, 0.08) 100%);
        backdrop-filter: blur(10px);
        border-left: 5px solid;
        border-image: linear-gradient(180deg, #667eea, #764ba2, #f093fb) 1;
        padding: 25px;
        border-radius: 18px;
        margin: 18px 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-top-right-radius: 18px;
        border-bottom-right-radius: 18px;
        animation: slideInRight 0.6s ease-out;
    }
    
    .recommendation-card:hover {
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transform: translateX(8px);
        background: linear-gradient(135deg, rgba(245, 247, 250, 0.12) 0%, rgba(195, 207, 226, 0.12) 100%);
    }
    
    .rec-name {
        font-size: 1.35em;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
    }
    
    .rec-score {
        font-size: 2.2em;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 12px 0;
        letter-spacing: -1px;
    }
    
    .badge-excellent {
        display: inline-block;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 10px 18px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.95em;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
        letter-spacing: 0.5px;
    }
    
    .badge-strong {
        display: inline-block;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 10px 18px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.95em;
        box-shadow: 0 4px 15px rgba(240, 87, 108, 0.3);
        letter-spacing: 0.5px;
    }
    
    .badge-good {
        display: inline-block;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 10px 18px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.95em;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        letter-spacing: 0.5px;
    }
    
    /* Favorite Item */
    .favorite-item {
        background: linear-gradient(135deg, rgba(255, 236, 210, 0.12) 0%, rgba(252, 182, 159, 0.12) 100%);
        backdrop-filter: blur(10px);
        padding: 18px 25px;
        border-radius: 15px;
        margin: 12px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 15px rgba(252, 182, 159, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: slideInRight 0.6s ease-out;
    }
    
    .favorite-item:hover {
        box-shadow: 0 8px 25px rgba(252, 182, 159, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
        background: linear-gradient(135deg, rgba(255, 236, 210, 0.18) 0%, rgba(252, 182, 159, 0.18) 100%);
    }
    
    /* Section Title */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 2em;
        font-weight: 800;
        color: #ffffff;
        margin: 40px 0 25px 0;
        padding-bottom: 18px;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2, #f093fb) 1;
        display: inline-block;
        animation: fadeInUp 0.8s ease-out;
        letter-spacing: -0.5px;
    }
    
    /* Search Container */
    .search-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 35px;
        border-radius: 20px;
        margin: 25px 0;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        animation: fadeInUp 0.8s ease-out 0.3s both;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .search-container h2 {
        color: white;
        margin-bottom: 20px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
        color: white !important;
        border: none !important;
        padding: 14px 32px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
        font-size: 1em !important;
        letter-spacing: 0.5px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.5) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    /* History Item */
    .history-item {
        background: linear-gradient(135deg, rgba(224, 195, 252, 0.15) 0%, rgba(142, 197, 252, 0.15) 100%);
        backdrop-filter: blur(10px);
        padding: 14px 20px;
        border-radius: 12px;
        margin: 10px 0;
        font-weight: 500;
        color: #e0c3fc;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 15px rgba(142, 197, 252, 0.1);
        animation: fadeInUp 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        background: linear-gradient(135deg, rgba(224, 195, 252, 0.25) 0%, rgba(142, 197, 252, 0.25) 100%);
        box-shadow: 0 6px 20px rgba(142, 197, 252, 0.2);
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 50px 30px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        color: #b0b8d4;
        border: 2px dashed rgba(255, 255, 255, 0.2);
        animation: fadeInUp 0.8s ease-out;
    }
    
    .empty-state h3 {
        color: #ffffff;
        font-size: 1.4em;
        margin-bottom: 10px;
        font-weight: 700;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 40px;
        color: #b0b8d4;
        font-size: 1em;
        margin-top: 60px;
        border-top: 2px solid rgba(255, 255, 255, 0.1);
        animation: fadeInUp 0.8s ease-out 0.5s both;
    }
    
    .footer p {
        margin: 8px 0;
        letter-spacing: 0.5px;
    }
    
    /* Selectbox & Slider */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb) !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
        margin: 30px 0 !important;
    }
    
    /* Info/Warning/Error boxes */
    .stInfo {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.15) 100%) !important;
        border: 1px solid rgba(79, 172, 254, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(17, 153, 142, 0.15) 0%, rgba(56, 239, 125, 0.15) 100%) !important;
        border: 1px solid rgba(17, 153, 142, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.15) 0%, rgba(245, 87, 108, 0.15) 100%) !important;
        border: 1px solid rgba(240, 147, 251, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(245, 87, 108, 0.15) 0%, rgba(240, 147, 251, 0.15) 100%) !important;
        border: 1px solid rgba(245, 87, 108, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
    }
</style>
""", unsafe_allow_html=True)

FILE_ID = "1J8yk4T40xYfI7DbHOkLYrs8IL0oasKe-"

@st.cache_data
def load_data():
    if not os.path.exists("restaurants.pkl"):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", "restaurants_reduced (1)", quiet=False)
    return pickle.load(open("restaurants_reduced (1)","rb"))

df = load_data().copy()

# Verify required columns exist
required_columns = ["name", "combined_features"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

@st.cache_resource
def build_model(data):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    vectors = tfidf.fit_transform(data["combined_features"].fillna(""))
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(vectors)
    return vectors, nn

vectors, model = build_model(df)

if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Settings & Analytics")
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("🍽️ Total Restaurants", f"{len(df):,}")
    st.metric("❤️ Favorites Saved", len(st.session_state.favorites))
    st.metric("🔍 Searches Made", len(st.session_state.history))
    
    st.markdown("---")
    st.markdown("### 📚 About This App")
    st.info("🤖 **AI Restaurant Discovery** is an advanced recommendation engine powered by NLP and Machine Learning. Find restaurants similar to your favorites in seconds! Built for recruiters", icon="ℹ️")

# Developer Badge at the top
st.markdown("""
<div class='developer-badge'>
    <p><span class='emoji'>👨‍💻</span>Built with ❤️ by <a href='https://github.com/Sarveyasha23-coder' target='_blank'>Sarveyasha Sodhiya</a></p>
</div>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class='main-header'>
    <h1>🍽️ AI Restaurant Discovery</h1>
    <p>Discover Your Next Favorite Restaurant with AI-Powered Recommendations</p>
</div>
""", unsafe_allow_html=True)

# Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>📍 Total Restaurants</div>
        <div class='number'>{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>🎯 Unique Names</div>
        <div class='number'>{df["name"].nunique():,}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>❤️ Saved Favorites</div>
        <div class='number'>{len(st.session_state.favorites)}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>🔍 Total Searches</div>
        <div class='number'>{len(st.session_state.history)}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Search Section
st.markdown("""
<div class='search-container'>
    <h2>🔍 Find Your Next Favorite Restaurant</h2>
</div>
""", unsafe_allow_html=True)

col_search1, col_search2 = st.columns([3, 1])

with col_search1:
    restaurant = st.selectbox(
        "Select a restaurant to get recommendations:",
        sorted(df["name"].unique()),
        key="restaurant_select",
        help="Choose any restaurant you like to find similar ones!"
    )

with col_search2:
    num_recommendations = st.slider(
        "Results",
        min_value=1,
        max_value=20,
        value=10,
        help="Number of recommendations"
    )

def recommend(name, n_recs):
    try:
        idx = df[df["name"] == name].index[0]
        distances, indices = model.kneighbors(vectors[idx], n_neighbors=min(n_recs + 1, len(df)))
        recs = []
        for i, d in zip(indices.flatten()[1:], distances.flatten()[1:]):
            recs.append({
                "name": df.iloc[i]["name"],
                "score": round((1-d)*100, 2)
            })
        return recs
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return []

col_rec1, col_rec2, col_rec3 = st.columns([1, 1, 1])

with col_rec1:
    get_recs = st.button("🚀 Get Recommendations", use_container_width=True, key="get_recs")

with col_rec2:
    clear_history = st.button("🗑️ Clear History", use_container_width=True, key="clear_hist")

with col_rec3:
    clear_favorites = st.button("💔 Clear Favorites", use_container_width=True, key="clear_favs")

if clear_history:
    st.session_state.history = []
    st.success("✅ History cleared!")
    st.rerun()

if clear_favorites:
    st.session_state.favorites = []
    st.success("✅ Favorites cleared!")
    st.rerun()

if get_recs:
    st.session_state.history.append(restaurant)
    recommendations = recommend(restaurant, num_recommendations)
    
    if recommendations:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); color: white; padding: 25px; border-radius: 18px; margin: 25px 0; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3); border: 1px solid rgba(255, 255, 255, 0.2);'>
            <h2 style='margin: 0 0 8px 0; font-family: Playfair Display, serif; font-size: 1.8em;'>✨ Top Matches for <strong>{restaurant}</strong></h2>
            <p style='margin: 0; opacity: 0.9; font-size: 0.95em;'>🎯 Based on comprehensive NLP analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for recommendations
        rec_cols = st.columns(2)
        
        for idx, r in enumerate(recommendations):
            col = rec_cols[idx % 2]
            
            with col:
                if r["score"] >= 90:
                    badge = '<span class="badge-excellent">🟢 Excellent Match</span>'
                elif r["score"] >= 75:
                    badge = '<span class="badge-strong">🔥 Strong Match</span>'
                else:
                    badge = '<span class="badge-good">🔵 Good Match</span>'
                
                st.markdown(f"""
                <div class='recommendation-card'>
                    <div class='rec-name'>🏪 {r['name']}</div>
                    <div class='rec-score'>{r['score']}%</div>
                    <div>{badge}</div>
                </div>
                """, unsafe_allow_html=True)
                
                col_fav, col_info = st.columns([1, 1])
                with col_fav:
                    safe_key = f"add_{idx}_{hash(r['name']) % 100000}"
                    if st.button("❤️ Add to Favorites", key=safe_key, use_container_width=True):
                        if r["name"] not in st.session_state.favorites:
                            st.session_state.favorites.append(r["name"])
                            st.success(f"✅ Added to favorites!")
                            st.rerun()
                        else:
                            st.warning("⚠️ Already in favorites!")
    else:
        st.markdown("""
        <div class='empty-state'>
            <h3>😕 No recommendations found</h3>
            <p>Try selecting a different restaurant</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Favorites Section
st.markdown("<h2 class='section-title'>❤️ Your Favorites</h2>", unsafe_allow_html=True)

if st.session_state.favorites:
    for fav_idx, fav in enumerate(st.session_state.favorites):
        col_fav1, col_fav2 = st.columns([4, 1])
        with col_fav1:
            st.markdown(f"""
            <div class='favorite-item'>
                <span>🏪 <strong>{fav}</strong></span>
            </div>
            """, unsafe_allow_html=True)
        with col_fav2:
            safe_remove_key = f"remove_{fav_idx}_{hash(fav) % 100000}"
            if st.button("Remove", key=safe_remove_key, use_container_width=True):
                st.session_state.favorites.remove(fav)
                st.success("✅ Removed from favorites!")
                st.rerun()
else:
    st.markdown("""
    <div class='empty-state'>
        <h3>💔 No favorites yet</h3>
        <p>Search for restaurants and add them to your favorites to get started!</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Analytics Section
st.markdown("<h2 class='section-title'>📊 Analytics & Insights</h2>", unsafe_allow_html=True)

col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    sample_recs = []
    restaurants_to_sample = min(20, len(df))
    for restaurant_name in df["name"].unique()[:restaurants_to_sample]:
        try:
            recs = recommend(restaurant_name, 5)
            sample_recs.extend([r["score"] for r in recs])
        except:
            continue
    
    if sample_recs:
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=sample_recs,
            nbinsx=30,
            name="Match Score",
            marker=dict(color=sample_recs, colorscale="Viridis", showscale=False)
        ))
        fig1.update_layout(
            title="Distribution of Recommendation Scores",
            xaxis_title="Match Score (%)",
            yaxis_title="Frequency",
            showlegend=False,
            hovermode="x",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Generate recommendations to see analytics!")

with col_viz2:
    if st.session_state.history:
        history_counts = pd.Series(st.session_state.history).value_counts().head(10)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=history_counts.index,
            x=history_counts.values,
            orientation="h",
            marker=dict(color=history_counts.values, colorscale="Turbo", showscale=False)
        ))
        fig2.update_layout(
            title="Most Searched Restaurants",
            xaxis_title="Search Count",
            yaxis_title="Restaurant",
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("👉 Search for restaurants to see your analytics!")

st.markdown("---")

# Search History Section
st.markdown("<h2 class='section-title'>🕒 Search History</h2>", unsafe_allow_html=True)

if st.session_state.history:
    recent_searches = list(dict.fromkeys(st.session_state.history[::-1]))[:12]
    
    col_hist1, col_hist2, col_hist3 = st.columns(3)
    
    for idx, search in enumerate(recent_searches):
        if idx < 4:
            with col_hist1:
                st.markdown(f"<div class='history-item'>🔍 {search}</div>", unsafe_allow_html=True)
        elif idx < 8:
            with col_hist2:
                st.markdown(f"<div class='history-item'>🔍 {search}</div>", unsafe_allow_html=True)
        else:
            with col_hist3:
                st.markdown(f"<div class='history-item'>🔍 {search}</div>", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='empty-state'>
        <h3>📋 No search history yet</h3>
        <p>Your search history will appear here</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Premium Footer
st.markdown("""
<div class='footer'>
    <p style='font-size: 1.1em;'>🍽️ <strong>AI Restaurant Discovery Platform</strong></p>
    <p>Powered by Machine Learning & Natural Language Processing</p>
    <p style='color: #667eea; font-weight: 600; margin-top: 12px;'>Built with ❤️ by Sarveyasha Sodhiya</p>
    <p style='font-size: 0.9em; margin-top: 15px; color: #666;'>© 2026 • All Rights Reserved • Premium UI Experience</p>
</div>
""", unsafe_allow_html=True)
