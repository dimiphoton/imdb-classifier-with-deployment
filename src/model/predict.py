"""
predict.py

This script provides functions for making predictions using a saved model and tokenizer.
The user can choose between a trained model or an untrained model.

Example usage:
python predict.py --model trained --text "This is a movie overview"

python predict.py --model untrained --text "This is a movie overview"

"""

import argparse
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from src.data.constants import MODEL_PATH

# Define argument parser
parser = argparse.ArgumentParser(description='Make predictions using a saved DistilBERT model.')
parser.add_argument('--model', type=str, choices=['trained', 'untrained'], required=True, help='Choose between a trained model or an untrained model.')
parser.add_argument('--text', type=str, required=True, help='Enter a movie overview for prediction.')

# Define types for the function
def predict_genre(model_type: str, text: str) -> List[Tuple[str, float]]:
    """
    This function loads a saved DistilBERT model and tokenizer and makes predictions
    for the given movie overview. The function returns a list of genre names and their
    corresponding probabilities.

    Args:
    - model_type (str): Type of model to use ('trained' or 'untrained').
    - text (str): The movie overview to predict the genre of.

    Returns:
    - List of tuples (genre_name, probability).
    """
    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Load the model
    if model_type == 'trained':
        model = TFDistilBertForSequenceClassification.from_pretrained(f"{MODEL_PATH}/trained_models/movie_genre_classifier")
    elif model_type == 'untrained':
        model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=19)

    # Clean the text
    text = clean_text(text)

    # Encode the text
    encoding = tokenizer([text], max_length=200, truncation=True, padding=True)
    ds = tf.data.Dataset.from_tensor_slices(dict(encoding))

    # Make the prediction
    predictions = model.predict(ds)

    # Format the prediction output
    probas = np.exp(predictions[0]) / np.exp(predictions[0]).sum(-1)
    results = list(zip(model.config.id2label.values(), probas[0]))

    return results


def clean_text(text: str) -> str:
    """
    This function takes a movie overview text and returns a cleaned version.

    Args:
    - text (str): The movie overview to clean.

    Returns:
    - A cleaned version of the text.
    """
    if type(text) == float:
        text = str(text)
    text = text.lower()
    text = re.sub(r"(\b\w+)'s\b", r"\1", text)
    text = re.sub("[^a-zA-Z'-]", " ", text)
    words = word_tokenize(text)
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parser.parse_args()

    # Make the prediction
    results = predict_genre(args.model, args.text)

    # Print the results
    for genre, prob in results:
        print(f"{genre}: {prob}")    
