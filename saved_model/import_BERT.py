from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

model_name = "distilbert-base-uncased"
model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)