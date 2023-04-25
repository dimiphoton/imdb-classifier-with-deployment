import spacy

# Load the spaCy model for English language text
nlp = spacy.load('en_core_web_sm')

def extract_keywords(doc, num_keywords=5):
    """
    Extract the top n (default 5) keywords from a spaCy Doc object.
    """
    # Filter out stop words and punctuation
    filtered_words = [token for token in doc if not token.is_stop and not token.is_punct]

    # Get the frequency of each word
    word_freq = {}
    for word in filtered_words:
        if word.text not in word_freq:
            word_freq[word.text] = 1
        else:
            word_freq[word.text] += 1

    # Sort the words by frequency and return the top n
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word[0] for word in sorted_words[:num_keywords]]
