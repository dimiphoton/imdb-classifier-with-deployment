"""
pipeline.py

This script is the main pipeline for the movie genre prediction task.
It trains and evaluates a DistilBERT model using a dataset containing movie names, genre lists, and overviews.
The pipeline is designed to be flexible and can be modified to suit different tasks or model architectures.

The pipeline consists of the following steps:
1. Load and preprocess the dataset
2. Train the model
3. Evaluate the model
4. Visualize the model performance
5. Save the model and tokenizer

To fine-tune the model, make predictions, or generate plots, refer to the corresponding scripts.

The pipeline configuration can be customized by modifying the 'config.json' file in the project root directory.
"""

import json
import pandas as pd
import numpy as np
import ast
from typing import Dict, Any, Tuple
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from constants import DATA_INTERIM_PATH, DATA_FINAL_PATH
from data_loader import load_csv_data, clean_text, prepare_dataset
from fine_tune import fine_tune_model
from save_model import save_model_and_tokenizer
from plots import plot_history


def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration parameters from a JSON file.

    Args:
        file_path (str): The path to the configuration JSON file.

    Returns:
        A dictionary containing the configuration parameters.
    """
    with open(file_path, "r") as file:
        config = json.load(file)
    return config


# Main code

if __name__ == "__main__":
    # Load and preprocess the dataset
    df = prepare_dataset()

    # Train the model
    model, tokenizer, history = fine_tune_model(df)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_dataset.batch(BATCH_SIZE))
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Visualize the model performance
    plot_history(history)

    # Save the model and tokenizer
    save_model_and_tokenizer(model, tokenizer)

    # Load the embeddings
    embeddings = {}
    with open('./embeddings.csv', 'r') as f:
        for line in f:
            movie, embedding = line.strip().split(',')
            embedding = np.array([float(x) for x in embedding.split()])
            embeddings[movie] = embedding

    # Prompt the user for input and recommend movies
    print("Enter 5 movie overviews you like:")
    overviews = [clean_text(input()) for _ in range(5)]
    recommendations = recommend_movies(overviews)

    # Print the recommendations
    print("Recommended movies:")
    for movie in recommendations:
        print(movie)
