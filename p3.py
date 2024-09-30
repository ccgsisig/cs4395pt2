import numpy as np
from numpy import dot
from numpy.linalg import norm

def load_glove_model(file_path):
    """
    Load the GloVe model from a file.
    
    Args:
        file_path (str): Path to the GloVe embeddings file.
    
    Returns:
        dict: A dictionary mapping words to their embedding vectors.
    """
    print("Loading GloVe Model")
    glove_model = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            vector = np.array(split_line[1:], dtype=np.float32)
            glove_model[word] = vector
    print(f"{len(glove_model)} words loaded!")
    return glove_model

def cosine_similarity(word1, word2, glove_vectors):
    """
    Compute the cosine similarity between two words using their GloVe vectors.
    
    Args:
        word1 (str): First word.
        word2 (str): Second word.
        glove_vectors (dict): Dictionary of GloVe vectors.
    
    Returns:
        float or None: Cosine similarity between word1 and word2, or None if a word is not found.
    """
    if word1 in glove_vectors and word2 in glove_vectors:
        v1 = glove_vectors[word1]
        v2 = glove_vectors[word2]
        return dot(v1, v2) / (norm(v1) * norm(v2))
    return None

def find_most_similar(word, glove_vectors, top_n=5):
    """
    Find the top-N most similar words to a given word using cosine similarity.
    
    Args:
        word (str): The word to find similar words for.
        glove_vectors (dict): Dictionary of GloVe vectors.
        top_n (int): Number of most similar words to return.
    
    Returns:
        list or None: List of tuples (word, similarity) of the top-N most similar words, or None if the word is not found.
    """
    if word not in glove_vectors:
        return None
    
    vector = glove_vectors[word]
    similarities = {}
    
    for other_word, other_vector in glove_vectors.items():
        if other_word != word:  # Exclude the word itself
            similarity = cosine_similarity(word, other_word, glove_vectors)
            similarities[other_word] = similarity
            
    # Sort the words based on similarity
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_similarities[:top_n]

if __name__ == '__main__':
    # Load GloVe vectors
    glove_vectors = load_glove_model('glove.6B/glove.6B.50d.txt')
    
    # Compute cosine similarity for the specified word pairs
    pairs = [("cat", "dog"), ("car", "bus"), ("apple", "banana")]
    for word1, word2 in pairs:
        similarity = cosine_similarity(word1, word2, glove_vectors)
        if similarity is not None:
            print(f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}")
        else:
            print(f"One of the words '{word1}' or '{word2}' is not in the vocabulary.")
    
    # Find the top 5 most similar words for specified words
    words_to_check = ["king", "computer", "university"]
    for word in words_to_check:
        similar_words = find_most_similar(word, glove_vectors)
        if similar_words is not None:
            print(f"\nTop 5 most similar words to '{word}':")
            for similar_word, similarity in similar_words:
                print(f"{similar_word}: {similarity:.4f}")
        else:
            print(f"The word '{word}' is not in the vocabulary.")

"""
Answer the following questions based on the outputs of your program:

cosine similarity between 'cat' and 'dog': 
[YOUR ANSWER]
cosine similarity between 'car' and 'bus': 
[YOUR ANSWER]
cosine similarity between 'apple' and 'banana': 
[YOUR ANSWER]

top 5 most similar words to 'king':
[YOUR ANSWER]
top 5 most similar words to 'computer':
[YOUR ANSWER]
top 5 most similar words to 'university':
[YOUR ANSWER]
"""
