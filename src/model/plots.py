"""
plots.py

This module provides functions to generate performance plots for a machine learning model.

The following functions are included:
- plot_history: Plot the accuracy and loss of the training and validation sets over time.

"""

from typing import Dict, List, Any, Tuple
import tensorflow as tf
from matplotlib import pyplot as plt


def plot_history(history: tf.keras.callbacks.History) -> None:
    """Plot the accuracy and loss of the training and validation sets over time.

    Args:
    history: A History object from the Keras fit() method containing training and validation accuracy and loss.

    Returns:
    None
    """
    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs[0].plot(history.history['accuracy'], label='training set')
    axs[0].plot(history.history['val_accuracy'], label = 'validation set')
    axs[0].set(xlabel = 'Epoch', ylabel='Accuracy', ylim=[0, 1])

    axs[1].plot(history.history['loss'], label='training set')
    axs[1].plot(history.history['val_loss'], label = 'validation set')
    axs[1].set(xlabel = 'Epoch', ylabel='Loss', ylim=[0, 10])
    
    axs[0].legend(loc='lower right')
    axs[1].legend(loc='lower right')
