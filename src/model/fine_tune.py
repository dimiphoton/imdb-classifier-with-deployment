"""
fine_tune.py

This script contains functions to fine-tune a DistilBERT model for movie genre prediction.
It includes the following functions:

1. create_datasets: Create the train, validation, and test datasets.
2. build_model: Build the DistilBERT model.
3. fine_tune_model: Fine-tune the model using the train and validation datasets.
"""

import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from typing import Tuple

def create_datasets(df: pd.DataFrame, tokenizer: DistilBertTokenizer) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Create train, validation, and test datasets from a preprocessed DataFrame.

    Args:
        df: The preprocessed DataFrame.
        tokenizer: The DistilBertTokenizer.

    Returns:
        A tuple of train, validation, and test datasets.
    """

    # Your code to create datasets goes here

    return train_dataset, val_dataset, test_dataset


def build_model(num_labels: int) -> TFDistilBertForSequenceClassification:
    """
    Build the DistilBERT model for sequence classification.

    Args:
        num_labels: The number of labels for classification.

    Returns:
        The TFDistilBertForSequenceClassification model.
    """

    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

    return model


def fine_tune_model(train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, num_labels: int, class_weights_dict: dict) -> Tuple[TFDistilBertForSequenceClassification, tf.keras.callbacks.History]:
    """
    Fine-tune the DistilBERT model using the train and validation datasets.

    Args:
        train_dataset: The train dataset.
        val_dataset: The validation dataset.
        num_labels: The number of labels for classification.
        class_weights_dict: The dictionary containing class weights.

    Returns:
        A tuple of the fine-tuned model and training history.
    """

    # Build the model
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

    # Compile the model
    OPTIMIZER =  tf.keras.optimizers.Adam(learning_rate=3e-5)
    LOSS = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    METRICS = ['accuracy']
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    # Set training parameters
    BATCH_SIZE = 8
    EPOCHS = 5

    # Fine-tune the model
    history = model.fit(
        train_dataset.batch(BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=val_dataset.batch(BATCH_SIZE),
        class_weight=class_weights_dict
    )

    return model, history

