import streamlit as st
import pandas as pd
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Restaurant Discovery",
    page_icon="🍽️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color:#0f172a;
}

.stApp{
    background: linear-gradient(
    135deg,
    #0f172a,
    #1e293b,
    #111827);
}

.hero{
    padding:30px;
    border-radius:25px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    text-align:center;
}

.big-title{
    font-size:52px;
    font-weight:800;
    color:white;
}

.sub-title{
    font-size:20px;
    color:#cbd5e1;
}

.card{
    background: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:20px;
    margin-bottom:15px;
    backdrop-filter: blur(10px);
}

.footer{
    text-align:center;
    padding:20px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pickle.load(open("restaurants.pkl","rb"))

df = load_data()

# ---------------- BUILD MODEL ----------------

@st.cache_resource
def build_model():

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = tfidf.fit_transform(
        df["combined_features"]
    )

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    model.fit(vectors)

    return tfidf, vectors, model

tfidf, vectors, model = build_model()

# ---------------- HERO ----------------

st.markdown("""
<div class='hero'>
<div class='big-title'>
🍽️ AI Restaurant Discovery Platform
</div>

<div class='sub-title'>
Discover restaurants through intelligent review analysis and recommendation technology
</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Smart Filters")

selected_restaurant = st.selectbox(
    "Select Restaurant",
    sorted(df["name"].unique())
)

# ---------------- RECOMMEND FUNCTION ----------------

def recommend(name):

    idx = df[df["name"] == name].index[0]

    distances, indices = model.kneighbors(
        vectors[idx],
        n_neighbors=11
    )

    results = []

    for i, score in zip(
        indices.flatten()[1:],
        distances.flatten()[1:]
    ):

        match = round((1-score)*100,2)

        results.append({
            "name":df.iloc[i]["name"],
            "rate":df.iloc[i]["rate"],
            "cuisines":df.iloc[i]["cuisines"],
            "match":match
        })

    return results

# ---------------- BUTTON ----------------

if st.button("🚀 Discover Similar Restaurants"):

    recommendations = recommend(
        selected_restaurant
    )

    st.subheader(
        "✨ Recommended Restaurants"
    )

    for item in recommendations:

        st.markdown(f"""
        <div class='card'>
        <h3>{item['name']}</h3>

        <p>⭐ Rating: {item['rate']}</p>

        <p>🍜 Cuisine: {item['cuisines']}</p>

        <p>🎯 Match Score: {item['match']}%</p>

        </div>
        """, unsafe_allow_html=True)

# ---------------- ANALYTICS ----------------

st.write("")
st.header("📊 Restaurant Analytics")

col1, col2 = st.columns(2)

with col1:

    cuisine_data = (
        df["cuisines"]
        .astype(str)
        .str.split(",")
        .explode()
        .value_counts()
        .head(10)
    )

    fig1 = px.bar(
        cuisine_data,
        title="Top Cuisines"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

with col2:

    ratings = (
        df["rate"]
        .astype(str)
        .str.replace("/5","")
    )

    fig2 = px.histogram(
        ratings,
        title="Rating Distribution"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ---------------- FOOTER ----------------

st.markdown("""
<div class='footer'>

Built with ❤️ by Sarveyasha Sodhiya

</div>
""", unsafe_allow_html=True)
