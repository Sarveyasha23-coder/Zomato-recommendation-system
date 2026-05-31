import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="AI Restaurant Discovery", 
    page_icon="🍽️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .main-header {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3em;
        margin-bottom: 10px;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2em;
        opacity: 0.95;
        letter-spacing: 1px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-card .number {
        font-size: 2.5em;
        font-weight: 800;
        margin: 10px 0;
    }
    
    .metric-card .label {
        font-size: 0.95em;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .recommendation-card:hover {
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transform: translateX(5px);
    }
    
    .rec-name {
        font-size: 1.3em;
        font-weight: 700;
        color: #333;
        margin-bottom: 10px;
    }
    
    .rec-score {
        font-size: 2em;
        font-weight: 800;
        color: #667eea;
        margin: 10px 0;
    }
    
    .badge-excellent {
        display: inline-block;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
    }
    
    .badge-strong {
        display: inline-block;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
    }
    
    .badge-good {
        display: inline-block;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
    }
    
    .favorite-item {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .favorite-item:hover {
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    
    .section-title {
        font-size: 1.8em;
        font-weight: 800;
        color: #333;
        margin: 30px 0 20px 0;
        padding-bottom: 15px;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    .search-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    .history-item {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        padding: 12px 18px;
        border-radius: 8px;
        margin: 8px 0;
        font-weight: 500;
        color: #333;
    }
    
    .empty-state {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        color: #666;
    }
    
    .footer {
        text-align: center;
        padding: 30px;
        color: #666;
        font-size: 0.95em;
        margin-top: 50px;
        border-top: 2px solid #eee;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.5em;
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
    st.markdown("### ⚙️ Settings")
    theme = st.toggle("🌙 Dark Mode", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("Total Restaurants", f"{len(df):,}")
    st.metric("Favorites Saved", len(st.session_state.favorites))
    st.metric("Searches Made", len(st.session_state.history))
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.info("🍽️ AI-powered restaurant discovery using Natural Language Processing and Machine Learning to find restaurants similar to your favorites.", icon="ℹ️")

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
    <h2 style='color: white; margin-bottom: 20px;'>🔍 Find Your Next Favorite</h2>
</div>
""", unsafe_allow_html=True)

col_search1, col_search2 = st.columns([3, 1])

with col_search1:
    restaurant = st.selectbox(
        "Select a restaurant:",
        sorted(df["name"].unique()),
        key="restaurant_select",
        help="Choose a restaurant you like, and we'll find similar ones!"
    )

with col_search2:
    num_recommendations = st.slider(
        "# of recommendations",
        min_value=1,
        max_value=20,
        value=10,
        help="How many similar restaurants to show"
    )

def recommend(name, n_recs):
    idx = df[df["name"] == name].index[0]
    distances, indices = model.kneighbors(vectors[idx], n_neighbors=n_recs + 1)
    recs = []
    for i, d in zip(indices.flatten()[1:], distances.flatten()[1:]):
        recs.append({
            "name": df.iloc[i]["name"],
            "score": round((1-d)*100, 2)
        })
    return recs

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
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; margin: 20px 0;'>
            <h2>✨ Recommendations similar to <strong>{restaurant}</strong></h2>
            <p>Based on menu, ambiance, and customer reviews analysis</p>
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
                    if st.button("❤️ Add to Favorites", key=f"add_{r['name']}", use_container_width=True):
                        if r["name"] not in st.session_state.favorites:
                            st.session_state.favorites.append(r["name"])
                            st.success(f"Added to favorites!")
                            st.rerun()
                        else:
                            st.warning("Already in favorites!")
    else:
        st.markdown("""
        <div class='empty-state'>
            <h3>😕 No recommendations found</h3>
            <p>Try selecting a different restaurant</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Favorites Section
col_fav_title = st.columns([1, 1])
with col_fav_title[0]:
    st.markdown("<h2 class='section-title'>❤️ Your Favorites</h2>", unsafe_allow_html=True)

if st.session_state.favorites:
    for fav in st.session_state.favorites:
        col_fav1, col_fav2 = st.columns([4, 1])
        with col_fav1:
            st.markdown(f"""
            <div class='favorite-item'>
                <span>🏪 <strong>{fav}</strong></span>
            </div>
            """, unsafe_allow_html=True)
        with col_fav2:
            if st.button("Remove", key=f"remove_{fav}", use_container_width=True):
                st.session_state.favorites.remove(fav)
                st.success("Removed from favorites!")
                st.rerun()
else:
    st.markdown("""
    <div class='empty-state'>
        <h3>💔 No favorites yet</h3>
        <p>Search for restaurants and add them to your favorites!</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Analytics Section
st.markdown("<h2 class='section-title'>📊 Analytics & Insights</h2>", unsafe_allow_html=True)

# Create visualizations
col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    # Distribution of recommendation scores
    sample_recs = []
    for restaurant_name in df["name"].unique()[:50]:
        try:
            recs = recommend(restaurant_name, 5)
            sample_recs.extend([r["score"] for r in recs])
        except:
            pass
    
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
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)

with col_viz2:
    # Top 10 most searched restaurants
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
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("👉 Search for restaurants to see analytics!")

st.markdown("---")

# Search History Section
st.markdown("<h2 class='section-title'>🕒 Search History</h2>", unsafe_allow_html=True)

if st.session_state.history:
    # Show last 10 searches in reverse order
    recent_searches = list(dict.fromkeys(st.session_state.history[::-1]))[:10]
    
    col_hist1, col_hist2, col_hist3 = st.columns(3)
    
    for idx, search in enumerate(recent_searches):
        if idx < 4:
            with col_hist1:
                st.markdown(f"<div class='history-item'>🔍 {search}</div>", unsafe_allow_html=True)
        elif idx < 7:
            with col_hist2:
                st.markdown(f"<div class='history-item'>🔍 {search}</div>", unsafe_allow_html=True)
        else:
            with col_hist3:
                st.markdown(f"<div class='history-item'>🔍 {search}</div>", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='empty-state'>
        <h3>📋 No search history yet</h3>
        <p>Your searches will appear here</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    <p>🍽️ <strong>AI Restaurant Discovery</strong> • Powered by Machine Learning & NLP</p>
    <p>Built with ❤️ by Sarveyasha Sodhiya</p>
    <p style='font-size: 0.85em; margin-top: 15px; color: #999;'>© 2026 • All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
