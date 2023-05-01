from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import pandas as pd

app = FastAPI()

# Load the Parquet file containing the TF-IDF values
tfidf_df = pd.read_parquet('tfidf.parquet')

# Initialize the TF-IDF vectorizer and fit it to the corpus of text data
vectorizer = TfidfVectorizer()
corpus = tfidf_df['text'].tolist()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Compute the cosine similarity matrix for all users
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define the API endpoints

# Lists all of the known Mastodon accounts in the data-set.
@app.get('/mstdn-nlp/api/v1/accounts/')
def get_accounts() -> List[Dict[str, str]]:
    users_list = tfidf_df[['username', 'id']].to_dict('records')
    return users_list

# Returns the TF-IDF matrix row for the given Mastodon user.
@app.get('/api/v1/tf-idf/user-ids/{user_id}')
def get_tfidf(user_id: str) -> Dict[str, float]:
    user_tfidf = tfidf_df[tfidf_df['username'] == user_id].iloc[0]['tfidf']
    return dict(zip(vectorizer.get_feature_names(), user_tfidf))

# Returns the 10 nearest neighbors, as measured by the cosine-distance between the user's TF-IDF matrix row.
@app.get('/api/v1/tf-idf/user-ids/{user_id}/neighbors')
def get_neighbors(user_id: str) -> List[Dict[str, str]]:
    user_index = tfidf_df[tfidf_df['username'] == user_id].index[0]
    user_cosine_similarities = cosine_sim_matrix[user_index]
    # Get the indices of the 10 nearest neighbors
    nearest_neighbors_indices = user_cosine_similarities.argsort()[:-11:-1]
    # Create a list of dictionaries for the nearest neighbors
    nearest_neighbors = []
    for index in nearest_neighbors_indices:
        if index != user_index:
            neighbor_username = tfidf_df.iloc[index]['username']
            neighbor_id = tfidf_df.iloc[index]['id']
            nearest_neighbors.append({"username": neighbor_username, "id": neighbor_id})
    return nearest_neighbors