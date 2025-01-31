import streamlit as st
from mastodon import Mastodon
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
from textblob import TextBlob 
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.decomposition import LatentDirichletAllocation 
import openai
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

mastodon = Mastodon(
    access_token='pytooter_usercred.secret', 
    api_base_url='https://mastodon.social' 
)

def clean_content(content):
    soup = BeautifulSoup(content, "html.parser")
    clean_text = soup.get_text(separator=' ', strip=True)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text.strip()


def fetch_trending_hashtags(limit=30):
    return mastodon.trending_tags(limit=limit)

def fetch_trending_statuses(limit=30):
    statuses = mastodon.trending_statuses(limit=limit)
    data = []
    for status in statuses:
        data.append({
            "content": clean_content(status['content']),
            "favourites_count": status['favourites_count'], 
            "boosts_count": status['reblogs_count'],        
            "replies_count": status['replies_count']   
        })
    return pd.DataFrame(data)

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

def analyze_sentiment(df):
    sentiments = []
    for content in df['content']:
        blob = TextBlob(content)
        sentiments.append(blob.sentiment.polarity)
    df['Sentiment'] = sentiments
    return df

def get_top_topics(statuses, n_topics=5):
    """
    Uses Groq's Llama 3.3-70b-versatile model to extract the top trending topics from a list of social media statuses.
    """
    # Initialize Groq client
    client = Groq(api_key="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")

    # Prepare the prompt
    prompt = f"""
    Below are multiple social media statuses. Identify the top {n_topics} topics that are currently trending. 
    Provide short and concise topic labels.
    
    Statuses:
    {statuses}

    Return only the topics in a numbered list.
    """

    # Send the request to Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an AI that identifies trending topics from text data."},
            {"role": "user", "content": prompt}
        ]
    )

    topics = response.choices[0].message.content.strip()
    return topics

def main():
    st.title("Mastodon Trending Dashboard")
    st.write("### Automatically Displaying Trending Data")

    st.write("#### Trending Hashtags")
    trending_hashtags = fetch_trending_hashtags()
    hashtag_df = pd.DataFrame(trending_hashtags, columns=['name', 'url'])
    st.table(hashtag_df[['name']])  
    st.caption("**Explanation:** This table shows the top  trending hashtags on Mastodon. These are the most popular hashtags being used across the platform.")

    st.write("#### Trending Statuses")
    trending_statuses = fetch_trending_statuses()
    st.table(trending_statuses[['content', 'favourites_count', 'boosts_count', 'replies_count']])
    st.caption("**Explanation:** This table displays the top 30 trending statuses, including their content, number of favorites, boosts, and replies.")

    st.write("#### Engagement Metrics for Trending Statuses")
    fig = px.bar(trending_statuses, x='content', y=['favourites_count', 'boosts_count', 'replies_count'],
                 title='Engagement Metrics', labels={'value': 'Count', 'variable': 'Metric', 'content': 'Status'})
    st.plotly_chart(fig)
    st.caption("**Explanation:** This bar chart visualizes the engagement metrics (favorites, boosts, and replies) for each trending status. Longer bars indicate higher engagement.")

    st.write("#### Word Cloud of Trending Statuses")
    all_content = " ".join(trending_statuses['content'].tolist())
    wordcloud = generate_wordcloud(all_content)
    st.pyplot(wordcloud)
    st.caption("**Explanation:** This word cloud shows the most frequently used words in the trending statuses. Larger words indicate higher frequency.")

    st.write("#### Sentiment Analysis of Trending Statuses")
    trending_statuses = analyze_sentiment(trending_statuses)
    st.write(trending_statuses[['content', 'Sentiment']])
    st.caption("**Explanation:** This table shows the sentiment polarity of each trending status. Positive values indicate positive sentiment, while negative values indicate negative sentiment.")

    fig = px.histogram(trending_statuses, x='Sentiment', title='Sentiment Distribution')
    st.plotly_chart(fig)
    st.caption("**Explanation:** This histogram shows the distribution of sentiment across trending statuses. It helps identify whether the overall sentiment is positive, neutral, or negative.")

    st.write("#### LLM-Based Topic Modeling")
    trending_texts = "\n".join(trending_statuses['content'].tolist())
    trending_topics = get_top_topics(trending_texts)
    st.write(trending_topics)
    st.caption("**Explanation:** This section uses a large language model (LLM) to identify the top 5 trending topics from the content of the trending statuses.")

if __name__ == "__main__":
    main()
