"""
data_preparation.py

This script contains functions for preprocessing the movie dataset. It performs the following tasks:

1. Extract genre names from the genre column
2. Clean the movie overview text
3. Remove rows with empty genre lists
4. Filter out less frequent genres
5. Split the dataset into training, validation, and test sets

All functions in this script are designed to be reusable and can be easily adapted for different datasets or preprocessing requirements.
"""

import pandas as pd
import numpy as np
import ast
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from typing import List, Tuple

# Download necessary resources for text processing
nltk.download('punkt')
nltk.download('stopwords')


def extract_genre_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract genre names from the genres column of the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the movie data.
        
    Returns:
        pd.DataFrame: The modified DataFrame with the genre names extracted.
    """
    # Convert string values to list of dictionaries
    df.genres = df.genres.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Extract genre names
    df.genres = df.genres.apply(lambda x: [genre['name'] for genre in x])

    return df


def clean_text(text: str) -> str:
    """
    Clean the movie overview text by removing stopwords, stemming words, and performing other preprocessing tasks.
    
    Args:
        text (str): The raw movie overview text.
        
    Returns:
        str: The cleaned movie overview text.
    """
    if type(text) == float:  # change type to text for some plot in dataset
        text = str(text)
    text = text.lower()  # Lowercase
    text = re.sub(r"(\b\w+)'s\b", r"\1", text)  # remove 's suffix
    text = re.sub("[^a-zA-Z'-]", " ", text)  # remove non-letter characters
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words if w not in stop_words]  # stem words
    return " ".join(words)


def filter_genres(df: pd.DataFrame, min_genre_count: int = 100) -> pd.DataFrame:
    """
    Filter out less frequent genres from the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the movie data.
        min_genre_count (int, optional): The minimum count of a genre to be retained. Defaults to 100.
        
    Returns:
        pd.DataFrame: The modified DataFrame with less frequent genres filtered out.
    """
    # Calculate genre counts
    genres_counts = df.explode('genres')['genres'].value_counts()
    genres_to_drop = genres_counts.loc[lambda g: g < min_genre_count].index.tolist()

    # Filter out rows with less frequent genres
    mask = df['genres'].apply(lambda x: not any(item in genres_to_drop for item in x))
    df = df[mask]

    return df


def prepare_dataset(file_path: str, min_genre_count: int = 100) -> pd.DataFrame:
    """
    Load the movie dataset and perform preprocessing tasks.
    
    Args:
        file_path (str): The path to the CSV file containing the movie data.
        min_genre_count (int, optional): The minimum count of a genre to be retained. Defaults to 100.
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
"""
    # Load dataset
    df = pd.read_csv(file_path)

    # Extract genre names
    df = extract_genre_names(df)

    # Select relevant columns
    df = df[['title', 'genres', 'overview', 'tagline']]

    # Remove rows where genres list is empty
    df = df[df["genres"].map(len) > 0]

    # Clean the overview column
    df["clean_overview"] = df['overview'].apply(clean_text)

    # Filter out less frequent genres
    df = filter_genres(df, min_genre_count)

    return df

       
