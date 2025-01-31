import streamlit as st

# Set page configuration
st.set_page_config(page_title="Mastodon Trending Analysis", layout="wide")

# Custom styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #1E90FF;
    }
    .sub-text {
        font-size: 18px;
        text-align: center;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<h1 class="main-title">Mastodon Trending Analysis</h1>', unsafe_allow_html=True)

# Subtitle
st.markdown('<p class="sub-text">Explore insights on trending hashtags, statuses, and links on Mastodon.</p>', unsafe_allow_html=True)

# Layout sections
st.markdown('<div class="section-box">', unsafe_allow_html=True)

st.write("### 🔍 Navigate the App:")
st.write(
    """
    - **📌 Search Page**: Gain insights on any topic of interest.
    - **📊 Trending Page**: View the latest insights on trending topics.
    """
)

st.markdown('</div>', unsafe_allow_html=True)

