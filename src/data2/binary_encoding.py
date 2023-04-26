#!/usr/bin/env python3
"""
This script performs binary encoding on the 'genre_list' column of a given DataFrame.
The script uses the sklearn library for encoding and expects the input DataFrame to have a
'genre_list' column containing lists of genres.
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def binary_encode_genre_list(df):
    """
    Perform binary encoding on the 'genre_list' column of the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'genre_list' column.

    Returns:
        pd.DataFrame: A new DataFrame with the binary-encoded 'genre_list' column.
    """
    mlb = MultiLabelBinarizer()
    encoded_genres = mlb.fit_transform(df['genre_list'].apply(eval))
    encoded_df = df.join(pd.DataFrame(encoded_genres, columns=mlb.classes_))
    return encoded_df


if __name__ == "__main__":
    # Load the cleaned_data DataFrame
    cleaned_data = pd.read_csv("./data/1.interim/cleaned_data.csv")

    # Perform binary encoding on the 'genre_list' column
    encoded_data = binary_encode_genre_list(cleaned_data)

    # Save the encoded DataFrame to a CSV file (optional)
    encoded_data.to_csv("./data/1.interim/encoded_data.csv", index=False)

