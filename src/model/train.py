"""
This script combines the fine-tuned transformer model and the additional feature
to train the final genre classifier model, and saves the trained model for deployment.
"""

import argparse
from feature_extraction import extract_keywords
from fine_tune_transformer import df, model, tokenizer, train_genre_classification_model

def main(args):
    if args.task == 'extract_keywords':
        overview = args.overview
        doc = nlp(overview)
        keywords = extract_keywords(doc, args.num_keywords)
        print(keywords)
    elif args.task == 'train_genre_classification_model':
        trained_model = train_genre_classification_model(df, model, tokenizer, batch_size=args.batch_size, num_epochs=args.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['extract_keywords', 'train_genre_classification_model'])
    parser.add_argument('--overview', help='The movie overview to extract keywords from')
    parser.add_argument('--num_keywords', type=int, default=5, help='The number of keywords to extract')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='The number of epochs to train for')
    args = parser.parse_args()
    main(args)
