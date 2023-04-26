"""
This script reads the cleaned dataframe and performs necessary preprocessing,
and adds features (binary encoding,keywords)
"""

import os
import pandas as pd
import spacy
from ..constants import DATA_INTERIM_PATH, DATA_FINAL_PATH

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def binary_encode_genres(df, column="genre_list"):
    """
    Perform binary encoding on the specified column with genres.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the genre data.
        column (str): The name of the column containing genre data.
    
    Returns:
        pd.DataFrame: A DataFrame with binary encoded genre columns.
    """
    # Create an empty set to store unique genres
    unique_genres = set()

    # Iterate through the genre lists, split the string and add genres to the set
    for genre_list_str in df[column]:
        genre_list = genre_list_str[1:-1].split(", ")
        unique_genres.update(genre_list)

    # Create binary encoded columns for each genre
    for genre in unique_genres:
        df[genre] = df[column].apply(lambda x: 1 if genre in x else 0)

    return df

def extract_keywords(text, n=5):
    """
    Extract the top n keywords from a given text using spaCy.
    
    Args:
        text (str): The input text.
        n (int, optional): The number of keywords to extract. Default is 5.
    
    Returns:
        list: A list of the top n keywords.
    """
    doc = nlp(text)
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
    top_keywords = keywords[:n]
    return top_keywords

def add_keyword_features(df, text_column, n=5):
    """
    Add keyword features to the DataFrame using the spaCy keyword extraction function.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing the text data.
        n (int, optional): The number of keywords to extract. Default is 5.
    
    Returns:
        pd.DataFrame: The DataFrame with added keyword columns.
    """
    for i in range(n):
        keyword_column = f"keyword_{i+1}"
        df[keyword_column] = df[text_column].apply(lambda x: extract_keywords(x, n)[i] if len(x) > 0 else "")

    return df

def main():
    # Load the cleaned_data DataFrame
    cleaned_data_path = os.path.join(DATA_INTERIM_PATH, "cleaned_data.csv")
    cleaned_data = pd.read_csv(cleaned_data_path)

    # Perform binary encoding on the 'genre_list' column
    binary_encoded_data = binary_encode_genres(cleaned_data)

    # Add keyword features from the 'title' column (or any other text column)
    processed_data = add_keyword_features(binary_encoded_data, "title")

    # Save the processed_data DataFrame to the DATA_FINAL_PATH folder
    processed_data_path = os.path.join(DATA_FINAL_PATH, "processed_data.csv")
    processed_data.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to '{processed_data_path}'")

if __name__ == "__main__":
    main()
