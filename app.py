
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
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", "restaurants_reduced.pkl", quiet=False)
    return pickle.load(open("restaurants_reduced.pkl","rb"))

df = load_data().copy()

df["rate"] = pd.to_numeric(df["rate"].astype(str).str.replace("/5","", regex=False), errors="coerce").fillna(0)

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

c1,c2,c3,c4 = st.columns(4)
c1.metric("Restaurants", f"{len(df):,}")
c2.metric("Cuisines", df["cuisines"].nunique())
c3.metric("Avg Rating", round(df["rate"].mean(),2))
c4.metric("Favorites", len(st.session_state.favorites))

all_cuisines = sorted(set(x.strip() for row in df["cuisines"].dropna() for x in str(row).split(",")))
selected_cuisine = st.sidebar.selectbox("Cuisine Filter", ["All"] + all_cuisines)

restaurant = st.selectbox("🔍 Search Restaurant", sorted(df["name"].unique()))

def recommend(name):
    idx = df[df["name"] == name].index[0]
    distances, indices = model.kneighbors(vectors[idx], n_neighbors=11)
    recs = []
    for i, d in zip(indices.flatten()[1:], distances.flatten()[1:]):
        recs.append({
            "name": df.iloc[i]["name"],
            "rate": df.iloc[i]["rate"],
            "cuisines": df.iloc[i]["cuisines"],
            "score": round((1-d)*100,2)
        })
    return recs

if st.button("🚀 Get Recommendations"):
    st.session_state.history.append(restaurant)
    for r in recommend(restaurant):
        badge = "🟢 Excellent" if r["score"] >= 90 else "🟡 Strong"
        st.markdown(f"""
        <div class='card'>
        <h3>{r['name']}</h3>
        ⭐ Rating: {r['rate']}<br>
        🍜 {r['cuisines']}<br>
        🎯 Match: {r['score']}% {badge}
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"❤️ Favorite {r['name']}", key=r['name']):
            if r["name"] not in st.session_state.favorites:
                st.session_state.favorites.append(r["name"])

st.subheader("🏆 Top Rated Restaurants")
top = df.sort_values("rate", ascending=False).head(10)
st.dataframe(top[["name","rate","cuisines"]], use_container_width=True)

st.subheader("📊 Analytics")
a,b = st.columns(2)

with a:
    cuisine_counts = df["cuisines"].astype(str).str.split(",").explode().str.strip().value_counts().head(10)
    st.plotly_chart(px.bar(cuisine_counts, title="Top Cuisines"), use_container_width=True)

with b:
    st.plotly_chart(px.histogram(df, x="rate", title="Ratings Distribution"), use_container_width=True)

st.subheader("⭐ Favorites")
st.write(st.session_state.favorites)

st.subheader("🕒 Recommendation History")
st.write(st.session_state.history)

st.markdown("<hr><center>Developed by Sarveyasha Sodhiya</center>", unsafe_allow_html=True)
