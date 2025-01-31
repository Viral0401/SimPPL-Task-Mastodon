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
import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import numpy as np

CUSTOM_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
    'wouldn', "wouldn't"
}

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

def clean_content(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word.lower() for word in words if word.isalpha() and word not in stopwords.words('english')]
    return " ".join(words)

def clean_text(content):
    soup = BeautifulSoup(content, "html.parser")
    clean_text = soup.get_text(separator=' ', strip=True)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text.strip()

from textblob import TextBlob
import matplotlib.pyplot as plt
import streamlit as st

def calculate_polarity(text):
    return TextBlob(text).sentiment.polarity

def fetch_real_time_posts(topic, iterations=5, delay=1):
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

def generate_wordcloud(text,query):
    text = re.sub(r'http\S+|https\S+', '', text)
    text = re.sub(r'\b' + re.escape(query) + r'\b', '', text, flags=re.IGNORECASE)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

def create_author_connection_graph(df):
    author_hashtags = defaultdict(set)
    G = nx.Graph()
    
    for _, row in df.iterrows():
        try:
            author = row['Author']
            tags = {tag.strip().lower() for tag in row['Hashtags']}
            author_hashtags[author].update(tags)
        except:
            continue
    
    authors = list(author_hashtags.keys())
    for a1, a2 in combinations(authors, 2):
        common_tags = author_hashtags[a1].intersection(author_hashtags[a2])
        if common_tags:
            G.add_edge(a1, a2, weight=len(common_tags), title=f"Shared tags: {', '.join(common_tags)}")
    
    for node in G.nodes():
        G.nodes[node]['size'] = 15
        G.nodes[node]['title'] = f"Author: {node}"
        
    return G


def preprocess_hashtags(hashtags):
    processed_hashtags = []
    for hashtag in hashtags:
        hashtag = re.sub(r'\W+', '', hashtag)
        word_tokens = hashtag.split()
        filtered_text = [w for w in word_tokens if w.lower() not in CUSTOM_STOPWORDS and w.isalpha()]
        processed_hashtags.extend(filtered_text)
    return processed_hashtags

import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

def create_network_graphs(df):
    # Preprocess hashtags
    df['Processed_Hashtags'] = df['Hashtags'].apply(preprocess_hashtags)

    # Create a graph based on hashtag co-occurrences
    G = nx.Graph()

    # Add nodes and edges based on hashtag co-occurrences
    for hashtags in df['Processed_Hashtags']:
        for i, hashtag1 in enumerate(hashtags):
            for j in range(i + 1, len(hashtags)):
                hashtag2 = hashtags[j]
                if G.has_edge(hashtag1, hashtag2):
                    G[hashtag1][hashtag2]['weight'] += 1
                else:
                    G.add_edge(hashtag1, hashtag2, weight=1)

    # Draw the graph with hashtag co-occurrences
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.3)
    node_sizes = [G.degree(node) * 200 for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="skyblue", font_size=5, font_weight="bold", edge_color="gray")
    plt.title("Network Graph of Hashtag Co-occurrences")

    # Render the graph in Streamlit
    st.pyplot(plt)  # This will display the plot correctly in Streamlit

def visualize_graph(graph, height="600px"):
    nt = Network(height=height, width="100%", bgcolor="#222222", font_color="white")
    nt.from_nx(graph)
    nt.toggle_hide_edges_on_drag(True)
    nt.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -80000,
          "springLength": 200
        },
        "minVelocity": 0.75
      }
    }
    """)
    return nt


st.title("Mastodon Hashtag Lookup & Visualization")
query = st.text_input("Enter a topic to search for posts:")

if st.button("Fetch Posts"):
    if query:
        with st.spinner(f"Fetching posts for {query}..."):
            df = fetch_real_time_posts(query, iterations=5, delay=5)
            if not df.empty:
                df['Polarity'] = df['Content'].apply(calculate_polarity)
                df['Sentiment'] = df['Polarity'].apply(classify_sentiment)
                st.subheader(f"Recent Posts for #{query}")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download data as CSV", data=csv, 
                                 file_name=f"{query}_posts.csv", mime='text/csv')
                
                # st.subheader("Word Cloud of Post Content")
                # generate_wordcloud(" ".join(df['Content'].dropna()), query)
                
                st.subheader("Hashtag Co-occurrence Network")
                hashtag_graph = create_network_graphs(df)
                sentiment_analysis_visualizations(df)
                generate_wordclouds(df, query)

            else:
                st.write("No posts found for this hashtag.")
