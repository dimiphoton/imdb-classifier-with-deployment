"""
recommend.py

This script recommends movies based on user input.
It takes a list of movie overviews as input and returns a list of recommended movies.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from constants import DATA_FINAL_PATH

# Load the dataset
df = pd.read_csv(DATA_FINAL_PATH)

# Load the embeddings
embeddings = {}
with open('./embeddings.csv', 'r') as f:
    for line in f:
        movie, embedding = line.strip().split(',')
        embedding = np.array([float(x) for x in embedding.split()])
        embeddings[movie] = embedding

# Define the recommender function
def recommend_movies(overviews, num_recommendations=10):
    # Compute the average embedding of the input overviews
    input_embedding = np.mean([embeddings[o] for o in overviews], axis=0)
    # Compute the cosine similarity between the input embedding and all movie embeddings
    sims = cosine_similarity([input_embedding], list(embeddings.values()))[0]
    # Sort the similarities in descending order and get the top N movies
    indices = np.argsort(sims)[::-1][:num_recommendations]
    recommendations = [(list(embeddings.keys())[i], sims[i
