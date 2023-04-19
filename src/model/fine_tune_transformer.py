"""
This script contains the code for fine-tuning a pre-trained transformer model
(e.g., BERT) for genre classification based on movie plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from typing import Tuple

df = pd.read_csv('movie_data.csv', index_col='original_title')
mlb = MultiLabelBinarizer()
df['genre_binary'] = mlb.fit_transform(df['genre_list'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))

def train_genre_classification_model(df: pd.DataFrame, model: BertForSequenceClassification, tokenizer: BertTokenizer, batch_size: int = 32, num_epochs: int = 10, output_dir: str = 'models') -> Tuple[BertForSequenceClassification, pd.DataFrame]:
    """
    Fine-tunes a BERT model to classify movies according to their genre based on their plot.
    
    Args:
        df (pd.DataFrame): The movie data as a pandas DataFrame.
        model (BertForSequenceClassification): The pre-trained BERT model to fine-tune.
        tokenizer (BertTokenizer): The BERT tokenizer.
        batch_size (int, optional): The batch size for training. Defaults to 32.
        num_epochs (int, optional): The number of epochs to train for. Defaults to 10.
        output_dir (str, optional): The directory to save the trained model to. Defaults to 'models'.
    
    Returns:
        Tuple[BertForSequenceClassification, pd.DataFrame]: The fine-tuned BERT model and a DataFrame of the training performance.
    """
    X = df['overview'].values
    y = df['genre_binary'].values
    X_tokenized = tokenizer.batch_encode_plus(X, max_length=512, padding=True, truncation=True, return_tensors='pt')
    dataset = torch.utils.data.TensorDataset(X_tokenized['input_ids'], X_tokenized['attention_mask'], torch.tensor(y))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    training_performance = pd.DataFrame(columns=['epoch', 'loss'])
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (input_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fn(outputs.logits, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print('Epoch %d loss: %.3f' % (epoch+1, epoch_loss))
        training_performance = training_performance.append({'epoch': epoch+1, 'loss': epoch_loss}, ignore_index=True)

    if output_dir is not None:
        output_path = os.path.join(output_dir, 'fine_tuned_model.pth')
        torch.save(model.state_dict(), output_path)
        print('Model saved to', output_path)

    return model, training_performance


def plot_training_performance(training_performance: pd.DataFrame) -> None:
    """
    Plots the training performance over time.
    
    Args:
        training_performance (pd.DataFrame): A DataFrame containing the training performance, with columns "epoch" and "loss".
    """
    plt.plot(training_performance['epoch'], training_performance['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Performance')
    plt.show()
