import streamlit as st
import pandas as pd
import altair as alt
import joblib
from googleapiclient.discovery import build
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import re
import emoji
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from googleapiclient.errors import HttpError
import joblib


# Load the SVM model from the pickle file
svm_pipeline = joblib.load(open("svm_text_sentiment.pkl", "rb"))

# Function to predict sentiment using the loaded SVM model
def predict_sentiment(text):
    prediction = svm_pipeline.predict([text])[0]
    return prediction

# Function to preprocess text
def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    # Remove special characters, emojis, and unnecessary symbols
    text = re.sub(r'[^\w\s]', '', text)
    # Remove emojis
    text = emoji.demojize(text)
    # Eliminate URLs or hyperlinks
    text = re.sub(r'http\S+', '', text)
    # Normalize text (convert to lowercase)
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def fetch_comments(video_id, youtube_api_key):
    """
    Fetches comments for a YouTube video.
    """
    # Initialize an empty list to store comments
    all_comments = []

    try:
        # Build the YouTube service
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)

        # Fetch comments in batches until the quota is exhausted
        next_page_token = None
        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100,  # Adjust this value to fetch more comments per page
                pageToken=next_page_token
            )
            response = request.execute()

            # Extract comments from the response
            comments = [item['snippet']['topLevelComment']['snippet'] for item in response['items']]
            all_comments.extend(comments)

            # Check if there are more pages of comments
            if 'nextPageToken' in response:
                next_page_token = response['nextPageToken']
            else:
                break

        return all_comments

    except HttpError as e:
        st.error(f"Error fetching comments: {e}")
        return None

def main(smartphone_features, smartphone_keywords):
    st.title("Smartphone YouTube Comment Sentiment Analyzer")
    st.subheader("Search by Keyword or Video Link")

    # Add option to select search method
    search_method = st.radio("Search Method:", ("Search by Keyword", "Search by Video Link"))

    # Allow users to input their YouTube Data API key
    youtube_api_key = st.text_input("Enter your YouTube Data API Key")

    if search_method == "Search by Keyword":
        search_query = st.text_input("Search Query")

        if st.button("Search"):
            if not any(keyword in search_query.lower() for keyword in smartphone_keywords) and not any(keyword in search_query.lower() for keyword in smartphone_features):
                st.error("Please enter a smartphone-related keyword or feature.")
                return

            if not youtube_api_key:
                st.error("Please enter your YouTube Data API key.")
                return

            # Build the YouTube service
            youtube = build('youtube', 'v3', developerKey=youtube_api_key)

            # Search for videos related to the search query
            search_response = youtube.search().list(
                q=search_query,
                part='id,snippet',
                type='video',
                order='viewCount',
                maxResults=1  # Adjust this value as needed
            ).execute()

            # Filter videos based on view count (> 50k views) and keyword in title
            video_ids = []
            for item in search_response['items']:
                video_id = item['id']['videoId']
                video_stats = youtube.videos().list(
                    part='statistics',
                    id=video_id
                ).execute()
                view_count = int(video_stats['items'][0]['statistics']['viewCount'])
                video_title = item['snippet']['title'].lower()
                if view_count > 50000 and (any(keyword in video_title for keyword in smartphone_keywords) or any(keyword in video_title for keyword in smartphone_features)):
                    video_ids.append(video_id)

            # Initialize variables to store comments
            all_comments = []

            # Fetch comments for each filtered video
            for video_id in video_ids:
                try:
                    # Fetch comments in batches until the quota is exhausted
                    next_page_token = None
                    while True:
                        request = youtube.commentThreads().list(
                            part="snippet",
                            videoId=video_id,
                            textFormat="plainText",
                            maxResults=20,  # Adjust this value to fetch more comments per page
                            pageToken=next_page_token
                        )
                        response = request.execute()

                        # Extract comments from the response
                        comments = [item['snippet']['topLevelComment']['snippet'] for item in response['items']]
                        all_comments.extend(comments)

                        # Check if there are more pages of comments
                        if 'nextPageToken' in response:
                            next_page_token = response['nextPageToken']
                        else:
                            break
                except HttpError as e:
                    st.warning(f"Comments for video with ID {video_id} are disabled. Skipping...")
                    continue

            num_comments = len(all_comments)

            # Display the number of comments
            st.text(f"Number of Comments: {num_comments}")

            # Analyze sentiment for each comment using the predict_sentiment function
            sentiments = [predict_sentiment(preprocess_text(comment['textDisplay'])) for comment in all_comments]

            # Count positive, negative, and neutral sentiments
            sentiment_counts = Counter(sentiments)

            # Calculate percentage for each sentiment
            total_sentiments = sum(sentiment_counts.values())
            sentiment_percentages = {sentiment: count / total_sentiments * 100 for sentiment, count in sentiment_counts.items()}

            # Visualize sentiment distribution using a bar chart with percentages
            st.subheader("Sentiment Distribution")
            sentiment_df = pd.DataFrame({'Sentiment': list(sentiment_percentages.keys()), 'Percentage': list(sentiment_percentages.values())})
            bar_chart = alt.Chart(sentiment_df).mark_bar().encode(
                x='Sentiment',
                y='Percentage',
                color=alt.Color('Sentiment', scale=alt.Scale(domain=['positive', 'neutral', 'negative'], range=['green', 'orange', 'red'])),
                tooltip=['Sentiment', 'Percentage']
            ).properties(
                width=500,
                height=300
            )
            st.altair_chart(bar_chart, use_container_width=True)

            # Generate word cloud
            st.subheader("Word Cloud")
            comments_to_display = [comment['textDisplay'] for comment in all_comments]
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(comments_to_display))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

            # Separate comments into strongly positive and strongly negative lists based on sentiment and whether they mention smartphone features
            strongly_positive_comments_with_features = []
            strongly_negative_comments_with_features = []

            for comment, sentiment in zip(all_comments, sentiments):
                text = comment['textDisplay'].lower()
                # Check if the comment mentions any smartphone features
                if any(feature in text for feature in smartphone_features):
                    # Check if the comment contains keywords related to smartphones
                    if any(keyword in text for keyword in smartphone_keywords):
                        if sentiment == 'positive':
                            strongly_positive_comments_with_features.append(comment)
                        elif sentiment == 'negative':
                            strongly_negative_comments_with_features.append(comment)

            # Sort comments based on likes in descending order
            strongly_positive_comments_with_features.sort(key=lambda x: int(x['likeCount']), reverse=True)
            strongly_negative_comments_with_features.sort(key=lambda x: int(x['likeCount']), reverse=True)

            # Display the top 5 positive comments mentioning smartphone features
            st.subheader("Top 5 Positive Comments Mentioning Smartphone Features")
            for i, comment in enumerate(strongly_positive_comments_with_features[:5]):
                st.write(f"**Comment {i+1} (Likes: {comment['likeCount']})**: {comment['textDisplay']}")

            # Display the top 5 negative comments mentioning smartphone features
            st.subheader("Top 5 Negative Comments Mentioning Smartphone Features")
            for i, comment in enumerate(strongly_negative_comments_with_features[:5]):
                st.write(f"**Comment {i+1} (Likes: {comment['likeCount']})**: {comment['textDisplay']}")

            # Extract top 20 smartphone features and visualize them
            st.subheader("Top 20 Smartphone Features")
            features = [comment['textDisplay'] for comment in all_comments]
            word_vectorizer = CountVectorizer(stop_words='english', max_features=20)
            word_frequencies = word_vectorizer.fit_transform(features)
            feature_names = word_vectorizer.get_feature_names_out()
            feature_counts = word_frequencies.toarray().sum(axis=0)
            feature_df = pd.DataFrame({'Feature': feature_names, 'Count': feature_counts})

            # Visualize top 20 smartphone features with different colors
            bar_chart_features = alt.Chart(feature_df).mark_bar().encode(
                x='Feature',
                y='Count',
                color=alt.Color('Feature', scale=alt.Scale(scheme='set1')),
                tooltip=['Feature', 'Count']
            ).properties(
                width=800,
                height=400
            )
            st.altair_chart(bar_chart_features, use_container_width=True)

    elif search_method == "Search by Video Link":
        video_link = st.text_input("Enter YouTube Video Link")

        if st.button("Fetch Comments"):
            if not youtube_api_key:
                st.error("Please enter your YouTube Data API key.")
                return

            if not video_link:
                st.error("Please enter a valid YouTube video link.")
                return

            # Extract video ID from the link
            video_id = extract_video_id(video_link)

            if video_id:
                # Fetch video title
                video_title = fetch_video_title(video_id, youtube_api_key)

                if video_title:
                    st.title(video_title)
                    # Fetch comments for the video
                    comments = fetch_comments(video_id, youtube_api_key)

                    # Display the fetched comments
                    if comments:
                        st.text(f"Number of Comments: {len(comments)}")

                        # Analyze sentiment for each comment using the predict_sentiment function
                        sentiments = [predict_sentiment(preprocess_text(comment['textDisplay'])) for comment in comments]

                        # Count positive, negative, and neutral sentiments
                        sentiment_counts = Counter(sentiments)

                        # Calculate percentage for each sentiment
                        total_sentiments = sum(sentiment_counts.values())
                        sentiment_percentages = {sentiment: count / total_sentiments * 100 for sentiment, count in sentiment_counts.items()}

                        # Visualize sentiment distribution using a bar chart with percentages
                        st.subheader("Sentiment Distribution")
                        sentiment_df = pd.DataFrame({'Sentiment': list(sentiment_percentages.keys()), 'Percentage': list(sentiment_percentages.values())})
                        bar_chart = alt.Chart(sentiment_df).mark_bar().encode(
                            x='Sentiment',
                            y='Percentage',
                            color=alt.Color('Sentiment', scale=alt.Scale(domain=['positive', 'neutral', 'negative'], range=['green', 'orange', 'red'])),
                            tooltip=['Sentiment', 'Percentage']
                        ).properties(
                            width=500,
                            height=300
                        )
                        st.altair_chart(bar_chart, use_container_width=True)

                        # Generate word cloud
                        st.subheader("Word Cloud")
                        comments_to_display = [comment['textDisplay'] for comment in comments]
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(comments_to_display))
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        st.pyplot(plt)

                        # Separate comments into strongly positive and strongly negative lists based on sentiment and whether they mention smartphone features
                        strongly_positive_comments_with_features = []
                        strongly_negative_comments_with_features = []

                        for comment, sentiment in zip(comments, sentiments):
                            text = comment['textDisplay'].lower()
                            # Check if the comment mentions any smartphone features
                            if any(feature in text for feature in smartphone_features):
                                # Check if the comment contains keywords related to smartphones
                                if any(keyword in text for keyword in smartphone_keywords):
                                    if sentiment == 'positive':
                                        strongly_positive_comments_with_features.append(comment)
                                    elif sentiment == 'negative':
                                        strongly_negative_comments_with_features.append(comment)

                        # Sort comments based on likes in descending order
                        strongly_positive_comments_with_features.sort(key=lambda x: int(x['likeCount']), reverse=True)
                        strongly_negative_comments_with_features.sort(key=lambda x: int(x['likeCount']), reverse=True)

                        # Display the top 5 positive comments mentioning smartphone features
                        st.subheader("Top 5 Positive Comments Mentioning Smartphone Features")
                        for i, comment in enumerate(strongly_positive_comments_with_features[:5]):
                            st.write(f"**Comment {i+1} (Likes: {comment['likeCount']})**: {comment['textDisplay']}")

                        # Display the top 5 negative comments mentioning smartphone features
                        st.subheader("Top 5 Negative Comments Mentioning Smartphone Features")
                        for i, comment in enumerate(strongly_negative_comments_with_features[:5]):
                            st.write(f"**Comment {i+1} (Likes: {comment['likeCount']})**: {comment['textDisplay']}")

                        # Extract top 20 smartphone features and visualize them
                        st.subheader("Top 20 Smartphone Features")
                        features = [comment['textDisplay'] for comment in comments]
                        word_vectorizer = CountVectorizer(stop_words='english', max_features=20)
                        word_frequencies = word_vectorizer.fit_transform(features)
                        feature_names = word_vectorizer.get_feature_names_out()
                        feature_counts = word_frequencies.toarray().sum(axis=0)
                        feature_df = pd.DataFrame({'Feature': feature_names, 'Count': feature_counts})

                        # Visualize top 20 smartphone features with different colors
                        bar_chart_features = alt.Chart(feature_df).mark_bar().encode(
                            x='Feature',
                            y='Count',
                            color=alt.Color('Feature', scale=alt.Scale(scheme='set1')),
                            tooltip=['Feature', 'Count']
                        ).properties(
                            width=800,
                            height=400
                        )
                        st.altair_chart(bar_chart_features, use_container_width=True)

                    else:
                        st.warning("No comments found for the provided video link.")
                else:
                    st.error("Failed to fetch video title.")
            else:
                st.error("Invalid YouTube video link. Please enter a valid link.")

def extract_video_id(video_link):
    """
    Extracts video ID from a YouTube video link.
    """
    # Extract video ID from various formats of YouTube links
    patterns = [
        r"(?<=v=)[\w-]+",  # Extract from ?v= parameter
        r"(?<=be/)[\w-]+",  # Extract from /embed/ or /watch?v= URL
        r"([\w-]+)$"  # Extract from the last segment of URL
    ]
    for pattern in patterns:
        match = re.search(pattern, video_link)
        if match:
            return match.group()
    return None

def fetch_video_title(video_id, youtube_api_key):
    """
    Fetches the title of a YouTube video.
    """
    try:
        # Build the YouTube service
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)

        # Fetch video details
        video_response = youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()

        # Extract video title
        video_title = video_response['items'][0]['snippet']['title']
        return video_title

    except HttpError as e:
        st.error(f"Error fetching video title: {e}")
        return None

if __name__ == "__main__":
    smartphone_features = ["camera", "battery", "display", "performance", "storage", "RAM", "processor", "screen", "resolution", "design", "waterproof", "wireless charging", "fast charging"]
    smartphone_keywords = ["smartphone", "iphone", "android", "samsung", "galaxy", "google pixel", "huawei", "xiaomi", "oneplus", "motorola", "lg", "oppo", "vivo", "realme", "nokia"]
    main(smartphone_features, smartphone_keywords)