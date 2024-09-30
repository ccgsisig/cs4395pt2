import nltk
from nltk import bigrams
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
from collections import defaultdict, Counter

# Download required NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('brown')

# Preprocess the corpus: Tokenize, lowercase, and add start/end tokens
def preprocess(corpus):
    tokenized_corpus = []
    for sentence in corpus:
        # Tokenize and lowercase the sentence
        tokens = [word.lower() for word in sentence]
        # Add '<s>' at the start and '</s>' at the end of the sentence
        tokenized_corpus.append(['<s>'] + tokens + ['</s>'])
    return tokenized_corpus

# Build the bigram model: Create frequency distributions for unigrams and bigrams
def build_bigram_model(tokenized_corpus):
    bigram_freq = defaultdict(Counter)
    unigram_freq = Counter()
    
    for document in tokenized_corpus:
        unigram_freq.update(document)
        for w1, w2 in bigrams(document):
            bigram_freq[w1][w2] += 1
            
    return bigram_freq, unigram_freq

# Calculate bigram probability with optional smoothing
def bigram_probability(bigram_freq, unigram_freq, word1, word2, smoothing=False):
    V = len(unigram_freq)
    if smoothing:
        return (bigram_freq[word1][word2] + 1) / (unigram_freq[word1] + V)
    else:
        if unigram_freq[word1] == 0:
            return 0.0
        return bigram_freq[word1][word2] / unigram_freq[word1]

# Compute the probability of a sentence
def sentence_probability(bigram_freq, unigram_freq, sentence, smoothing=False):
    tokens = word_tokenize(sentence.lower())
    tokens = ['<s>'] + tokens + ['</s>']
    probability = 1.0
    for w1, w2 in bigrams(tokens):
        probability *= bigram_probability(bigram_freq, unigram_freq, w1, w2, smoothing)
    return probability

# Predict the next N words given a sentence prefix
def predict_next_words(bigram_freq, unigram_freq, sentence_prefix, N, smoothing=False):
    tokens = word_tokenize(sentence_prefix.lower())
    current_word = tokens[-1]
    generated_words = []
    
    for _ in range(N):
        if current_word in bigram_freq:
            next_word = bigram_freq[current_word].most_common(1)[0][0]
            if next_word == '</s>':
                break
            generated_words.append(next_word)
            current_word = next_word
        else:
            break
            
    return ' '.join(generated_words)

# Main execution
if __name__ == "__main__":
    # Load the corpus
    corpus = brown.sents()
    
    # Preprocess the corpus
    tokenized_corpus = preprocess(corpus)
    
    # Build the bigram model
    bigram_freq, unigram_freq = build_bigram_model(tokenized_corpus)
    
    # Calculate the probability of a test sentence
    test_sentence = "The dog barked at the cat."
    probability_no_smoothing = sentence_probability(bigram_freq, unigram_freq, test_sentence, smoothing=False)
    print(f"Sentence probability without smoothing: {probability_no_smoothing}")
    
    probability_with_smoothing = sentence_probability(bigram_freq, unigram_freq, test_sentence, smoothing=True)
    print(f"Sentence probability with smoothing: {probability_with_smoothing}")
    
    # Predict the next N words
    sentence_prefix = "I won 200"
    N = 5
    predicted_words = predict_next_words(bigram_freq, unigram_freq, sentence_prefix, N, smoothing=True)
    print(f"Predicted next {N} words: {predicted_words}")

"""
Answer the following questions based on the outputs of your program:

1. Sentence probability without smoothing: 
[YOUR ANSWER]
2. Sentence probability with smoothing: 
[YOUR ANSWER]
3. Predicted next 5 words:
[YOUR ANSWER]
"""
