"""
save_model.py

This script contains functions for saving the trained model and tokenizer to a directory.

Functions:
    save_model_and_tokenizer(model, tokenizer, save_dir):
        Saves the trained model and tokenizer to the specified directory.

"""

from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import os

def save_model_and_tokenizer(model: TFDistilBertForSequenceClassification, tokenizer: DistilBertTokenizer, save_dir: str) -> None:
    """
    Saves the trained model and tokenizer to the specified directory.

    Args:
        model (TFDistilBertForSequenceClassification): The trained model to be saved.
        tokenizer (DistilBertTokenizer): The tokenizer used to preprocess the input data.
        save_dir (str): The directory to save the model and tokenizer. If the directory does not exist, it will be created.

    Returns:
        None
    """
    # Create the save directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the model and tokenizer to the directory
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
