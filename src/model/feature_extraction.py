"""
This script includes the implementation of an additional feature using spaCy,
such as keyword extraction, sentiment analysis, or text summarization.
"""

import spacy
from typing import List

nlp = spacy.load('en_core_web_sm')

def extract_keywords(doc: spacy.tokens.Doc, num_keywords: int) -> List[str]:
    """
    Extracts the top `num_keywords` keywords from a spaCy `Doc` object.
    
    Args:
        doc (spacy.tokens.Doc): The spaCy `Doc` object to extract keywords from.
        num_keywords (int): The number of keywords to extract.
    
    Returns:
        List[str]: A list of the top `num_keywords` keywords.
    """
    noun_chunks = list(doc.noun_chunks)
    nouns = [chunk.root.text for chunk in noun_chunks if chunk.root.pos_ == 'NOUN']
    adjectives = [token.text for token in doc if token.pos_ == 'ADJ']
    keywords = nouns + adjectives
    return list(set(keywords))[:num_keywords]
