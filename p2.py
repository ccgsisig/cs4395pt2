import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter
import math
import numpy as np

# Download the Brown corpus if not already downloaded
nltk.download('brown')

# Preprocess the corpus: Tokenize, lowercase, and add start/end tokens
def preprocess(corpus):
    tokenized_corpus = []
    for sentence in corpus:
        tokens = [word.lower() for word in sentence]  # Tokenize and lowercase
        tokenized_corpus.append(tokens)  # No need for <s> and </s> in this case
    return tokenized_corpus

# Calculate Term Frequency (TF)
def compute_tf(corpus):
    tf = defaultdict(Counter)
    for doc_index, document in enumerate(corpus):
        tf[doc_index] = Counter(document)  # Count words in the document
    return tf

# Calculate Document Frequency (DF)
def compute_df(tf):
    df = Counter()
    for document in tf.values():
        for word in set(document):  # Unique words only
            df[word] += 1
    return df

# Calculate TF-IDF for each word
def compute_tfidf(tf, df, num_docs):
    tfidf = defaultdict(dict)
    for doc_index, document in tf.items():
        for word, count in document.items():
            tfidf[doc_index][word] = count * math.log(num_docs / (1 + df[word]))
    return tfidf

# Create a word co-occurrence matrix
def create_cooccurrence_matrix(corpus, window_size=5):
    # Build the vocabulary of unique words
    vocab = set(word for document in corpus for word in document)
    word_to_id = {word: idx for idx, word in enumerate(vocab)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    
    # Initialize co-occurrence matrix
    cooccurrence_matrix = np.zeros((len(vocab), len(vocab)))

    # Fill in the co-occurrence matrix
    for document in corpus:
        for i, word in enumerate(document):
            word_id = word_to_id[word]
            left_context = document[max(0, i - window_size):i]
            right_context = document[i + 1:i + 1 + window_size]
            context_words = left_context + right_context
            for context_word in context_words:
                context_id = word_to_id[context_word]
                cooccurrence_matrix[word_id, context_id] += 1

    return cooccurrence_matrix, word_to_id, id_to_word

# Calculate PPMI from co-occurrence matrix
def compute_ppmi(cooccurrence_matrix):
    total_sum = np.sum(cooccurrence_matrix)
    word_occurrences = np.sum(cooccurrence_matrix, axis=1)
    
    ppmi_matrix = np.zeros(cooccurrence_matrix.shape)
    
    for i in range(cooccurrence_matrix.shape[0]):
        for j in range(cooccurrence_matrix.shape[1]):
            if cooccurrence_matrix[i, j] > 0:
                p_wi_wj = cooccurrence_matrix[i, j] / total_sum
                p_wi = word_occurrences[i] / total_sum
                p_wj = word_occurrences[j] / total_sum
                pmi = math.log(p_wi_wj / (p_wi * p_wj)) if (p_wi * p_wj) > 0 else 0
                ppmi_matrix[i, j] = max(0, pmi)  # Only keep positive values

    return ppmi_matrix

# Main execution
if __name__ == "__main__":
    # Load the Brown corpus as sentences
    corpus = brown.sents()[0:1000]  # Use first 1000 sentences

    # Preprocess the corpus
    processed_corpus = preprocess(corpus)

    # Number of documents in the corpus
    num_docs = len(processed_corpus)

    # Step 1: Calculate TF-IDF
    tf = compute_tf(processed_corpus)
    df = compute_df(tf)
    tfidf = compute_tfidf(tf, df, num_docs)

    # Output TF-IDF for a few words in the first document
    print("TF-IDF for word 'county' in the first document: ", tfidf[0].get('county', 0))
    print("TF-IDF for word 'investigation' in the first document: ", tfidf[0].get('investigation', 0))
    print("TF-IDF for word 'produced' in the first document: ", tfidf[0].get('produced', 0))

    # Step 2: Calculate PPMI
    window_size = 5
    cooccurrence_matrix, word_to_id, id_to_word = create_cooccurrence_matrix(processed_corpus, window_size=window_size)
    ppmi_matrix = compute_ppmi(cooccurrence_matrix)

    # Output PPMI for a few word pairs
    print("\nPPMI for a few word pairs:")
    words = [['expected', 'approve'], ['mentally', 'in'], ['send', 'bed']]
    for word_pair in words:
        word1, word2 = word_pair
        word1_id = word_to_id.get(word1, None)
        word2_id = word_to_id.get(word2, None)
        if word1_id is not None and word2_id is not None:
            ppmi_value = ppmi_matrix[word1_id, word2_id]
            print(f"PPMI({word1}, {word2}) = {ppmi_value}")
        else:
            print(f"Words '{word1}' or '{word2}' not found in vocabulary.")

"""
Answer the following questions based on the outputs of your program:

TF-IDF for word 'county' in the first document:  
[YOUR ANSWER]
TF-IDF for word 'investigation' in the first document:  
[YOUR ANSWER]
TF-IDF for word 'produced' in the first document: 
[YOUR ANSWER]

PPMI for a few word pairs:
PPMI(expected, approve) = 
[YOUR ANSWER]
PPMI(mentally, in) = 
[YOUR ANSWER]
PPMI(send, bed) = 
[YOUR ANSWER]
"""
