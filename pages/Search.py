import streamlit as st
from mastodon import Mastodon
import pandas as pd
import time
import networkx as nx
from pyvis.network import Network
from itertools import combinations
from collections import defaultdict
from wordcloud import WordCloud
import re
from bs4 import BeautifulSoup
import string
import matplotlib.pyplot as plt
from textblob import TextBlob

mastodon = Mastodon(
    access_token="pytooter_usercred",  
    api_base_url="https://mastodon.social"
)

def calculate_polarity(text):
    return TextBlob(text).sentiment.polarity

def classify_sentiment(polarity):
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def generate_wordclouds(df, query):
    positive_text = " ".join(df[df['Polarity'] > 0]['Content'])
    negative_text = " ".join(df[df['Polarity'] < 0]['Content'])
    
    positive_text = re.sub(r'http\S+|https\S+', '', positive_text)
    positive_text = re.sub(r'\b' + re.escape(query) + r'\b', '', positive_text, flags=re.IGNORECASE)
    
    negative_text = re.sub(r'http\S+|https\S+', '', negative_text)
    negative_text = re.sub(r'\b' + re.escape(query) + r'\b', '', negative_text, flags=re.IGNORECASE)

    positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    negative_wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)
    
    st.subheader("Word Cloud for Positive Sentiments")
    st.image(positive_wordcloud.to_array())

    st.subheader("Word Cloud for Negative Sentiments")
    st.image(negative_wordcloud.to_array())

def sentiment_analysis_visualizations(df):
    df['Polarity'] = df['Content'].apply(calculate_polarity)
    df['Sentiment'] = df['Polarity'].apply(classify_sentiment)

    if 'favourites_count' in df.columns:
        st.subheader("Polarity vs Likes")
        plt.figure(figsize=(10, 6))
        plt.scatter(df['favourites_count'], df['Polarity'], color='purple', alpha=0.5)
        plt.title("Polarity vs Number of Likes")
        plt.xlabel("Number of Likes")
        plt.ylabel("Polarity")
        st.pyplot(plt)
    else:
        st.warning("The dataset doesn't contain a 'favourites_count' column for the scatter plot.")

    sentiment_counts = df['Sentiment'].value_counts()
    st.subheader("Sentiment Distribution")
    plt.figure(figsize=(7, 7))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'grey', 'red'])
    plt.title("Distribution of Sentiment (Positive, Neutral, Negative)")
    st.pyplot(plt)

def clean_text(content):
    soup = BeautifulSoup(content, "html.parser")
    clean_text = soup.get_text(separator=' ', strip=True)
    clean_text = re.sub(r'http\S+', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = clean_text.translate(str.maketrans('', '', string.punctuation))
    return clean_text.strip()

def fetch_real_time_posts(topic, iterations=5, delay=5):
    hashtag = topic.replace(" ", "")
    all_data = []
    max_id = None

    for i in range(iterations):
        posts = mastodon.timeline_hashtag(hashtag, limit=100, max_id=max_id)
        if not posts:
            break
        for post in posts:
            mentions = [mention['acct'] for mention in post['mentions']]
            all_data.append({
                "Content": clean_text(post['content']),
                "Hashtags": [tag['name'] for tag in post['tags']],
                "Author": post['account']['username'],
                "Mentions": mentions,
                "favourites_count": post['favourites_count'], 
                "boosts_count": post['reblogs_count'],        
                "replies_count": post['replies_count']  
            })
        max_id = posts[-1]['id']
        if i < iterations - 1:
            time.sleep(delay)

    return pd.DataFrame(all_data)

def create_network_graphs(df):
    hashtag_pairs = []
    for hashtags in df['Hashtags']:
        if len(hashtags) > 1:
            hashtag_pairs.extend(combinations(hashtags, 2))
    
    G = nx.Graph()
    for pair in hashtag_pairs:
        if G.has_edge(*pair):
            G[pair[0]][pair[1]]['weight'] += 1
        else:
            G.add_edge(pair[0], pair[1], weight=1)

    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.5)
    node_sizes = [G.degree(node) * 100 for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    plt.title("Network Graph of Hashtag Co-occurrences")
    st.pyplot(plt)

def visualize_graph(graph, height="600px"):
    nt = Network(height=height, width="100%", bgcolor="#222222", font_color="white")
    nt.from_nx(graph)
    return nt

st.title("Mastodon Hashtag Lookup & Visualization")
query = st.text_input("Enter a topic to search for posts:")

if st.button("Fetch Posts"):
    if query:
        with st.spinner(f"Fetching posts for {query}..."):
            df = fetch_real_time_posts(query, iterations=5, delay=5)
            if not df.empty:
                st.subheader(f"Recent Posts for #{query}")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download data as CSV", data=csv, file_name=f"{query}_posts.csv", mime='text/csv')
                
                st.subheader("Word Cloud of Post Content")
                generate_wordclouds(df, query)
                
                st.subheader("Hashtag Co-occurrence Network")
                create_network_graphs(df)
                
                sentiment_analysis_visualizations(df)
            else:
                st.write("No posts found for this hashtag.")
