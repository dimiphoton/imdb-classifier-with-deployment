"""
data_loader.py

This script contains functions for loading and preparing the movie dataset for the movie genre prediction task.
It includes functions to load the CSV file, preprocess the text, and prepare the dataset for model training and evaluation.

Functions:
- load_csv_data: Load movie data from a CSV file.
- clean_text: Clean and preprocess the text in the overview column.
- prepare_dataset: Preprocess and prepare the dataset for model training and evaluation.
"""

import pandas as pd
import ast
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from typing import List, Tuple

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load movie data from a CSV file.

    Args:
        file_path (str): The file path of the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the movie data.
    """
    df = pd.read_csv(file_path)
    return df

def clean_text(text: str) -> str:
    """
    Clean and preprocess the text in the overview column.

    Args:
        text (str): The text to clean and preprocess.

    Returns:
        str: The cleaned and preprocessed text.
    """
    if type(text) == float:  # change type to text for some plot in the dataset
        text = str(text)
    
    text = text.lower()  # Lowercase
    text = re.sub(r"(\b\w+)'s\b", r"\1", text)  # remove 's suffix
    text = re.sub("[^a-zA-Z'-]", " ", text)  # remove non-letter characters

    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words if w not in stop_words]  # stem words
    
    return " ".join(words)

def prepare_dataset(file_path: str) -> pd.DataFrame:
    """
    Preprocess and prepare the dataset for model training and evaluation.

    Args:
        file_path (str): The file path of the CSV file.

    Returns:
        pd.DataFrame: A preprocessed DataFrame containing the movie data.
    """
    df = load_csv_data(file_path)

    # Convert string values to list of dictionaries
    df.genres = df.genres.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Extract genre names
    df.genres = df.genres.apply(lambda x: [genre['name'] for genre in x])

    df = df[['title', 'genres', 'overview', 'tagline']]

    # Remove rows where genres list is empty
    df = df[df["genres"].map(len) > 0]

    # Clean the overview column
    df["clean_overview"] = df['overview'].apply(clean_text)

    # Filter out rows with rare genres
    genres_counts = df.explode('genres')['genres'].value_counts()
    genres_to_drop = genres_counts.loc[lambda g: g < 100].index.tolist()
    mask = df['genres'].apply(lambda x: not any(item in genres_to_drop for item in x))
    df = df[mask]

    return df
