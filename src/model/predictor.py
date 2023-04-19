import pandas as pd
import spacy
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

# Load the movie data and pre-trained BERT tokenizer
df = pd.read_csv('movie_data.csv', index_col='original_title')
mlb = MultiLabelBinarizer()
df['genre_binary'] = mlb.fit_transform(df['genre_list'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the fine-tuned BERT model from file
model = BertForSequenceClassification.from_pretrained('models/fine_tuned_model.pth', num_labels=len(mlb.classes_))

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Define a function to predict the genres of a movie based on its description
def predict_genres(description: str) -> list:
    # Tokenize the description and classify the movie
    X = [description]
    X_tokenized = tokenizer.batch_encode_plus(X, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(X_tokenized['input_ids'], attention_mask=X_tokenized['attention_mask'])
    predicted_genres = mlb.inverse_transform((outputs.logits > 0).detach().cpu().numpy())[0]
    
    return predicted_genres

# Example usage
description = 'A young boy in a small town discovers a mysterious egg that hatches into a friendly dragon. But when a mean-spirited hunter sets out to capture the dragon, the boy and his new friend must evade the hunter and protect the dragon.'
predicted_genres = predict_genres(description)
print('Predicted genres:', predicted_genres)
