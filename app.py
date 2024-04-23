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
import nltk

# Download NLTK stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')  # Add this line to download WordNet data


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

# Function to map sentiment score to labels
def map_sentiment(score):
    if score > 0:  # Positive sentiment
        return 'Positive'
    elif score == 0:  # Neutral sentiment
        return 'Neutral'
    else:  # Negative sentiment
        return 'Negative'

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
                maxResults=10,  # Adjust this value to fetch more comments per page
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
    
    # Instructions Section
    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. **Enter Your YouTube Data API Key:**
        - Before you begin, ensure you have a YouTube Data API key. If you don't have one, you can obtain it from the Google Cloud Console.
        - Copy and paste your API key into the text input provided.
        
    2. **Search for Smartphone-related Videos:**
        - In the "Search Smartphone" field, enter keywords related to the smartphone videos you want to analyze.
        - Click the "Search" button to initiate the search.
        
    3. **Explore Sentiment Analysis:**
        - Once the search is complete, the app will fetch the most popular videos related to your query (with over 50,000 views).
        - The app will then analyze the sentiment of comments on these videos.
        - You will see a distribution of sentiments (positive, neutral, negative) in a bar chart.
        - Additionally, a word cloud will visualize the most common words used in the comments.
        
    4. **Explore Top Comments:**
        - The app identifies and displays the top 5 positive and negative comments mentioning smartphone features.
        - These comments are sorted based on the number of likes they received.
        
    5. **Explore Smartphone Features:**
        - The app extracts and visualizes the top 20 smartphone features mentioned in the comments.
        - Features are displayed in a bar chart, highlighting their frequency in the comments.
        
    6. **Note:**
        - This app requires a valid YouTube Data API key to fetch videos and comments.
        - Comments with disabled comment sections or comments containing less than 20 characters are excluded from analysis.
    """)
    st.subheader("Search by Keyword")

    # Allow users to input their YouTube Data API key
    youtube_api_key = st.text_input("Enter your YouTube Data API Key")

    search_query = st.text_input("Search Smartphone")

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
            maxResults=10  # Adjust this value as needed
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

        # Map sentiment scores to labels using the map_sentiment function
        sentiments_labels = [map_sentiment(score) for score in sentiment_percentages.keys()]

        # Create DataFrame with sentiment labels and corresponding percentages
        sentiment_df = pd.DataFrame({'Sentiment': sentiments_labels, 'Percentage': sentiment_percentages.values()})

        # Create the bar chart
        bar_chart = alt.Chart(sentiment_df).mark_bar().encode(
            x=alt.X('Sentiment', axis=alt.Axis(labels=True)),  # Allow axis labels
            y='Percentage',
            color=alt.Color('Sentiment', scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['green', 'orange', 'red'])),
            tooltip=['Sentiment', 'Percentage']
        ).properties(
            width=500,
            height=300
        )

        # Update the chart axis with labels
        bar_chart = bar_chart.configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_legend(
            titleFontSize=12,
            labelFontSize=10
        )

        # Display the chart
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
                    mapped_sentiment = map_sentiment(sentiment)
                    if mapped_sentiment == 'Positive':
                        strongly_positive_comments_with_features.append(comment)
                    elif mapped_sentiment == 'Negative':
                        strongly_negative_comments_with_features.append(comment)

        # Sort comments based on the number of likes (or any other engagement metric)
        strongly_positive_comments_with_features.sort(key=lambda x: int(x.get('likeCount', 0)), reverse=True)
        strongly_negative_comments_with_features.sort(key=lambda x: int(x.get('likeCount', 0)), reverse=True)

        # Display the top 5 positive comments mentioning smartphone features
        st.subheader("Top 5 Positive Comments Mentioning Smartphone Features")
        for i, comment in enumerate(strongly_positive_comments_with_features[:5]):
            st.write(f"**Comment {i+1} (Likes: {comment.get('likeCount', 0)})**: {comment['textDisplay']}")

        # Display the top 5 negative comments mentioning smartphone features
        st.subheader("Top 5 Negative Comments Mentioning Smartphone Features")
        for i, comment in enumerate(strongly_negative_comments_with_features[:5]):
            st.write(f"**Comment {i+1} (Likes: {comment.get('likeCount', 0)})**: {comment['textDisplay']}")


        # Extract top 10 smartphone features and visualize them
        st.subheader("Top 10 Smartphone Features")
        features = [comment['textDisplay'] for comment in all_comments]
        word_vectorizer = CountVectorizer(stop_words='english', max_features=10)
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


if __name__ == "__main__":
    smartphone_features = ["camera", "battery", "display", "performance", "storage", "RAM", "processor", "screen", "resolution", "design", "waterproof", "wireless charging", "fast charging"]
    smartphone_keywords = ["smartphone", "iphone", "android", "samsung", "galaxy", "google pixel", "huawei", "xiaomi", "oneplus", "motorola", "lg", "oppo", "vivo", "realme", "nokia"]
    main(smartphone_features, smartphone_keywords)
