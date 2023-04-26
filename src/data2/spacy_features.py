#!/usr/bin/env python3
"""
This script adds features to a given DataFrame using spaCy keyword extraction.
The script expects the input DataFrame to have a 'text' column containing text data.
"""

import pandas as pd
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")


def extract_keywords(text, n=10):
    """
    Extract keywords from a given text using spaCy.

    Args:
        text (str): The input text.
        n (int, optional): The number of keywords to extract. Default is 10.

    Returns:
        list: A list of the extracted keywords.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    pattern = [{"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}}]
    matcher.add("Keyword", [pattern])
    matches = matcher(doc)
    keywords = [doc[match[1]:match[2]].text for match in matches]
    return keywords[:n]


def add_spacy_features(df):
    """
    Add spaCy features (keywords) to the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'text' column.

    Returns:
        pd.DataFrame: A new DataFrame with the added spaCy features.
    """
    nlp = spacy.load("en_core_web_sm")
    df["keywords"] = df["text"].apply(extract_keywords)
    return df


if __name__ == "__main__":
    # Load the cleaned_data DataFrame
    cleaned_data = pd.read_csv("./data/1.interim/cleaned_data.csv")

    # Add features using spaCy keyword extraction
    spacy_data = add_spacy_features(cleaned_data)

    # Save the DataFrame with spaCy features to a CSV file
