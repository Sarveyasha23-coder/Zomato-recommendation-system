import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

st.set_page_config(page_title="AI Restaurant Discovery", page_icon="🍽️", layout="wide")

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

theme = st.sidebar.toggle("🌙 Dark Mode", value=True)

bg = "#0f172a" if theme else "#f8fafc"
text = "white" if theme else "#111827"
card = "#1e293b" if theme else "white"

st.markdown(f"""
<style>
.stApp{{background:{bg};color:{text};}}
.card{{background:{card};padding:18px;border-radius:18px;margin:10px 0;}}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🍽️ AI Restaurant Discovery Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>NLP Powered Restaurant Recommendation Engine</p>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Restaurants", f"{len(df):,}")
c2.metric("Unique Names", df["name"].nunique())
c3.metric("Total Records", len(df))
c4.metric("Favorites", len(st.session_state.favorites))

restaurant = st.selectbox("🔍 Search Restaurant", sorted(df["name"].unique()))

def recommend(name):
    idx = df[df["name"] == name].index[0]
    distances, indices = model.kneighbors(vectors[idx], n_neighbors=11)
    recs = []
    for i, d in zip(indices.flatten()[1:], distances.flatten()[1:]):
        recs.append({
            "name": df.iloc[i]["name"],
            "score": round((1-d)*100, 2)
        })
    return recs

if st.button("🚀 Get Recommendations"):
    st.session_state.history.append(restaurant)
    recommendations = recommend(restaurant)
    
    if recommendations:
        st.subheader(f"Recommendations similar to {restaurant}")
        for r in recommendations:
            badge = "🟢 Excellent Match" if r["score"] >= 90 else "🟡 Strong Match" if r["score"] >= 75 else "🔵 Good Match"
            st.markdown(f"""
            <div class='card'>
            <h3>{r['name']}</h3>
            🎯 Similarity Score: {r['score']}% {badge}
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"❤️ Add to Favorites", key=r['name']):
                if r["name"] not in st.session_state.favorites:
                    st.session_state.favorites.append(r["name"])
                    st.success(f"Added {r['name']} to favorites!")
    else:
        st.warning("No recommendations found.")

st.divider()

st.subheader("⭐ Your Favorites")
if st.session_state.favorites:
    for fav in st.session_state.favorites:
        col1, col2 = st.columns([4, 1])
        col1.write(f"🏷️ {fav}")
        if col2.button("Remove", key=f"remove_{fav}"):
            st.session_state.favorites.remove(fav)
            st.rerun()
else:
    st.info("No favorites yet. Start searching and add restaurants to your favorites!")

st.divider()

st.subheader("🕒 Recommendation History")
if st.session_state.history:
    history_df = pd.DataFrame({
        "Search Query": st.session_state.history
    })
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.info("No search history yet.")

st.markdown("<hr><center>Developed by Sarveyasha Sodhiya</center>", unsafe_allow_html=True)
