"""
embeddings.py

This script generates embeddings for each movie overview using a trained DistilBERT model.
The embeddings are saved to a file in the format "movie_name,embedding".
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from data_loader import clean_text
from constants import DATA_FINAL_PATH

# Load the saved model and tokenizer from a directory
model = TFDistilBertForSequenceClassification.from_pretrained('./saved_model/')
tokenizer = DistilBertTokenizer.from_pretrained('./saved_model/')

# Load the dataset
df = pd.read_csv(DATA_FINAL_PATH)

# Clean the overview column
df["clean_overview"] = df['overview'].apply(clean_text)

# Generate embeddings for each overview
embeddings = []
for i, row in df.iterrows():
    text = row['clean_overview']
    encodings = tokenizer([text], max_length=200, truncation=True, padding=True)
    ds = tf.data.Dataset.from_tensor_slices(dict(encodings))
    embedding = model.predict(ds)[0]
    embeddings.append((row['title'], embedding))

# Save the embeddings to a file
with open('./embeddings.csv', 'w') as f:
    for e in embeddings:
        movie = e[0]
        embedding = ",".join(str(x) for x in e[1])
        f.write(f"{movie},{embedding}\n")
