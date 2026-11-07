import numpy as np
import random
import emoji
import re
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util
import requests
import os
import pickle
import torch

def sample_users(tweets_data, user_descriptions, K, M1, desired_users, method="random"):
    """
    Sample users based on the specified method, including selecting recently active users.

    Parameters:
        tweets_data (dict): Dictionary containing user tweets.
        user_descriptions (dict): Dictionary containing user descriptions.
        K (int): Number of previous tweets to select.
        M1 (int): Number of future tweets to select.
        desired_users (int): Number of users to sample.
        method (str): Sampling method. Options are "random", "top_active", or "recent_activity".

    Returns:
        list: List of sampled user IDs.
    """
    print(f"Sampling users with at least {K + M1} tweets...")

    # Filter users with enough tweets
    valid_users = [uid for uid in tweets_data if len(tweets_data[uid]) >= K + M1]

    if not valid_users:
        raise ValueError("No users found with enough tweets for the specified K and M1.")

    if method == "random":
        # Random sampling
        sampled_users = random.sample(valid_users, min(desired_users, len(valid_users)))

    elif method == "top_active":
        # Top active users
        sorted_users = sorted(valid_users, key=lambda uid: len(tweets_data[uid]), reverse=True)
        sampled_users = sorted_users[:desired_users]

    elif method == "recent_activity":
        # Find the most recent tweet date in the dataset
        all_tweet_dates = []
        user_tweet_dates = {}

        for user_id, tweets in tweets_data.items():
            user_dates = []
            for tweet in tweets:
                try:
                    tweet_date = datetime.strptime(tweet["tweet_created_at"], "%a %b %d %H:%M:%S %z %Y")
                    user_dates.append(tweet_date)
                except (ValueError, KeyError, TypeError):
                    continue
            if user_dates:
                all_tweet_dates.extend(user_dates)
                user_tweet_dates[user_id] = user_dates

        if not all_tweet_dates:
            raise ValueError("No valid tweet dates found in the dataset.")

        # Determine the latest date in the dataset and calculate the cutoff date
        latest_date = max(all_tweet_dates)
        cutoff_date = latest_date - timedelta(weeks=3)

        # Identify users who have tweets in the recent activity range
        recently_active_users = [
            user_id for user_id, dates in user_tweet_dates.items()
            if any(date >= cutoff_date for date in dates)
        ]

        # Filter users who also satisfy the K + M1 tweet condition
        recent_valid_users = [uid for uid in recently_active_users if len(tweets_data[uid]) >= K + M1]

        if not recent_valid_users:
            raise ValueError("No users found with recent activity and enough tweets.")

        # Sort users by number of tweets and sample the top desired_users users
        sorted_recent_users = sorted(recent_valid_users, key=lambda uid: len(tweets_data[uid]), reverse=True)
        sampled_users = sorted_recent_users[:min(desired_users, len(sorted_recent_users))]

        if not sampled_users:
            print("Not enough users meet the desired conditions. Returning fewer users.")
    else:
        raise ValueError(f"Invalid sampling method: {method}. Choose 'random', 'top_active', or 'recent_activity'.")

    print(f"Sampled {len(sampled_users)} users using the '{method}' method.")
    return sampled_users

def expand_url(short_url):
    """
    Expands a shortened URL to its full format by following redirects.

    Args:
        short_url (str): Shortened URL.

    Returns:
        str: Expanded URL or the original URL if expansion fails.
    """
    try:
        response = requests.head(short_url, allow_redirects=True, timeout=5)
        return response.url
    except requests.RequestException:
        return short_url  # Return the original URL if expansion fails

def preprocess_tweets(tweets):
    """
    Preprocesses a list of tweets to decode Unicode sequences into emojis,
    remove noise, normalize text, and expand URLs.

    Args:
        tweets (list): List of tweet texts.

    Returns:
        list: List of preprocessed tweets.
    """
    preprocessed_tweets = []

    for tweet in tweets:
        # Ensure the tweet is a string
        tweet = str(tweet)

        # Decode Unicode sequences into emojis and normalize spaces
        normalized_tweet = re.sub(r'\s+', ' ', emoji.emojize(emoji.demojize(tweet)).strip())

        # Expand URLs in the tweet
        url_pattern = re.compile(r'https?://\S+')
        expanded_tweet = url_pattern.sub(lambda match: expand_url(match.group(0)), normalized_tweet)

        # Add the processed tweet to the list
        preprocessed_tweets.append(expanded_tweet)

    return preprocessed_tweets

def get_user_tweets(tweets_data, user_id, K, M1):
    """
    Samples the user's tweets for classification.
    Args:
        tweets_data (dict): Dictionary of all tweets grouped by user_id.
        user_id (int): The user ID whose tweets to sample.
        K (int): Number of previous tweets to select.
        M1 (int): Number of future tweets to select.
    Returns:
        tuple: Lists of previous tweets, future tweets (both as dictionaries), and their original tweet data.
    """
    user_tweets = tweets_data[user_id]
    # Separate previous and future tweets
    previous_tweets_data = user_tweets[-(K + M1):-M1]
    future_tweets_data = user_tweets[-M1:]

    # Preprocess tweets to expand URLs
    previous_tweets = [{"text": text, "tweet_created_at": tweet["tweet_created_at"]}
                       for tweet, text in zip(previous_tweets_data, preprocess_tweets([t["text"] for t in previous_tweets_data]))]
    future_tweets = [{"text": text, "tweet_created_at": tweet["tweet_created_at"]}
                     for tweet, text in zip(future_tweets_data, preprocess_tweets([t["text"] for t in future_tweets_data]))]

    return previous_tweets, future_tweets, future_tweets_data

def load_all_graphs(graph_folder):
    """Preloads all graph pickle files into memory to avoid repeated disk I/O."""
    graph_cache = {}
    for filename in os.listdir(graph_folder):
        if filename.startswith("gr_week_") and filename.endswith(".pkl"):
            round_num = filename.split("_")[-1].split(".")[0]
            graph_path = os.path.join(graph_folder, filename)
            try:
                with open(graph_path, "rb") as f:
                    graph_cache[round_num] = pickle.load(f)
                print(f"Graph {filename} loaded successfully.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                graph_cache[round_num] = None
    return graph_cache

GRAPH_CACHE = None  # loaded lazily by get_graph_cache()


def get_graph_cache():
    """Load the weekly follower graphs (gr_week_<round>.pkl) once, from $VAPOR_GRAPHS."""
    global GRAPH_CACHE
    if GRAPH_CACHE is None:
        GRAPH_CACHE = load_all_graphs(os.environ.get("VAPOR_GRAPHS", "data/raw/graphs"))
    return GRAPH_CACHE

def get_others_tweets(
    tweets_data, user_id, M2, method="random", user_tweets=None, max_candidates=1000000,
    batch_size=5000
):
    """
    Samples tweets from other users for classification based on the specified method.

    Args:
        tweets_data (dict): Dictionary of all tweets grouped by user_id.
        user_id (int): The user ID whose tweets to exclude.
        M2 (int): Number of tweets to sample from other users.
        method (str): Sampling method. Options are "random", "similar_topics", "followees", "followers", or "both".
        user_tweets (list): List of the user's tweets (required for some methods).
        max_candidates (int): Maximum number of candidate tweets for similarity computation.
        batch_size (int): Number of candidate tweets to process per batch for similarity computation.

    Returns:
        list: Preprocessed tweets from other users or an empty list if no tweets are found.
    """
    
    # Validate method
    if method not in ["random", "similar_topics", "followees", "followers", "both"]:
        raise ValueError(f"Invalid method: {method}. Choose 'random', 'similar_topics', 'followees', 'followers', or 'both'.")

    # Get all other users (excluding the specified user)
    other_users = [uid for uid in tweets_data if uid != user_id]
    all_other_tweets = [tweet for uid in other_users for tweet in tweets_data[uid]]

    # Determine timeframe for filtering tweets
    future_times = [
        datetime.strptime(tweet["tweet_created_at"], "%a %b %d %H:%M:%S %z %Y")
        for tweet in user_tweets
    ]
    min_time, max_time = min(future_times), max(future_times)

    # Filter tweets within timeframe
    time_filtered_tweets = [
        tweet for tweet in all_other_tweets
        if min_time <= datetime.strptime(tweet["tweet_created_at"], "%a %b %d %H:%M:%S %z %Y") <= max_time
    ]
    print(f"Total tweets from all users: {len(all_other_tweets)}")
    print(f"Tweets remaining after timeframe filter: {len(time_filtered_tweets)}")

    # Random sampling method
    if method == "random":
        return random.sample(time_filtered_tweets, min(M2, len(time_filtered_tweets)))

    # Similar topics method
    elif method == "similar_topics":
        if len(time_filtered_tweets) > max_candidates:
            time_filtered_tweets = random.sample(time_filtered_tweets, max_candidates)

        # Load embedding model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

        # Embed user tweets
        user_embeddings = model.encode([tweet["text"] for tweet in user_tweets], convert_to_tensor=True)

        # Compute similarity scores in batches
        avg_scores = []
        for i in range(0, len(time_filtered_tweets), batch_size):
            batch = time_filtered_tweets[i:i + batch_size]
            batch_texts = [tweet["text"] for tweet in batch]
            batch_embeddings = model.encode(batch_texts, convert_to_tensor=True)
            batch_similarity_scores = util.pytorch_cos_sim(user_embeddings, batch_embeddings)
            avg_scores.extend(batch_similarity_scores.mean(dim=0).cpu().numpy())

        # Rank tweets by similarity and select the top M2
        sorted_indices = sorted(range(len(avg_scores)), key=lambda k: avg_scores[k], reverse=True)
        selected_tweets = [time_filtered_tweets[i] for i in sorted_indices[:M2]]
        return preprocess_tweets([tweet["text"] for tweet in selected_tweets])

    # Followees, followers, and both methods
    elif method in ["followees", "followers", "both"]:
        rounds = list(set(tweet["Round"] for tweet in user_tweets if "Round" in tweet))

        followees, followers = set(), set()
        for round_num in rounds:
            graph = get_graph_cache().get(str(round_num))  # Access preloaded graph
            if graph:
                graph_user_id = str(user_id)
                if graph_user_id in graph:
                    print(f"User {graph_user_id} found in graph for round {round_num}.")
                    if method in ["followees", "both"]:
                        followees.update(graph.successors(graph_user_id))
                    if method in ["followers", "both"]:
                        followers.update(graph.predecessors(graph_user_id))
                else:
                    print(f"User {graph_user_id} not found in graph for round {round_num}.")

        # Filter valid user IDs
        valid_followees = {uid for uid in followees if uid.isdigit()}
        valid_followers = {uid for uid in followers if uid.isdigit()}

        # Determine target users based on method
        target_users = valid_followees if method == "followees" else valid_followers
        if method == "both":
            target_users = valid_followees & valid_followers

        # Collect tweets from target users
        target_tweets = [
            tweet for target_user_id in target_users
            for tweet in tweets_data.get(int(target_user_id), [])
        ]

        # Filter target tweets by timeframe
        time_filtered_target_tweets = [
            tweet for tweet in target_tweets
            if min_time <= datetime.strptime(tweet["tweet_created_at"], "%a %b %d %H:%M:%S %z %Y") <= max_time
        ]
        print(f"Total target tweets: {len(target_tweets)}")
        print(f"Target tweets after timeframe filter: {len(time_filtered_target_tweets)}")

        return preprocess_tweets([tweet["text"] for tweet in random.sample(
            time_filtered_target_tweets, min(M2, len(time_filtered_target_tweets))
        )])

    print(f"No valid tweets found for method: {method}")
    return []
