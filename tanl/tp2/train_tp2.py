from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string
import os
import gensim.downloader as api
from gensim.models import Word2Vec
import multiprocessing
import time


#################### COMMON CODE FOR BOTH FILES


def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence


def parse_input_file(file_path):
    processed_sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                sentence_id, sentence = parts
                preprocessed_sentence = preprocess_sentence(sentence)
                processed_sentences.append((sentence_id, preprocessed_sentence))
    return processed_sentences


def sentence_to_vector(sentence, model):
    words = sentence.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)


def compute_similarity_matrix(sentence_vectors):
    return cosine_similarity(sentence_vectors)


def find_most_similar_sentences(processed_sentences, model):
    sentence_vectors = [sentence_to_vector(sentence, model) for _, sentence in processed_sentences]
    similarity_matrix = compute_similarity_matrix(sentence_vectors)
    np.fill_diagonal(similarity_matrix, -1)  # to kill self similarity
    most_similar_sentences = {}
    for i, (sentence_id, _) in enumerate(processed_sentences):
        similar_sentence_idx = np.argmax(similarity_matrix[i])
        most_similar_sentence_id = processed_sentences[similar_sentence_idx][0]
        most_similar_sentences[sentence_id] = most_similar_sentence_id  # set the value in a dictionary
    return most_similar_sentences


#################### END COMMON CODE FOR BOTH FILES


def train_word2vec_model(corpus, vector_size=100, window=5, min_count=5, workers=multiprocessing.cpu_count()):
    """
    Train a Word2Vec model on the given corpus.
    :param corpus: Iterable of lists of words (each list represents a sentence)
    :param vector_size: Dimensionality of the word vectors
    :param window: Maximum distance between the current and predicted word within a sentence
    :param min_count: Ignores all words with total frequency lower than this
    :param workers: Number of worker threads to train the model
    :return: Trained Word2Vec model
    """
    model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

start_time = time.time()
# Load the dataset using Gensim's downloader API
dataset = api.load("text8")
model_file = "word2vec.model"
if os.path.isfile(model_file):
    model = Word2Vec.load(model_file)
else:
    dataset = api.load("text8")
    model = train_word2vec_model(dataset)
    model.save(model_file)

input_file_path = 'T_sent.txt'
# Make sure to preprocess your input data as before
processed_data = parse_input_file(input_file_path)

# Find most similar sentences using your trained model
similar_sentences = find_most_similar_sentences(processed_data, model)

# Save results to out2.2.txt
with open('out2.2.txt', 'w', encoding='utf-8') as f:
    for sentence_id, similar_id in similar_sentences.items():
        f.write(f'{sentence_id}\t{similar_id}\n')


# Generate sentence vectors
sentence_vectors = [sentence_to_vector(sentence, model) for _, sentence in processed_data]

# Filter out sentences that couldn't be vectorized (i.e., resulted in zero vectors)
sentence_vectors = [vec for vec in sentence_vectors if not np.all(vec == 0)]

# Convert to a NumPy array
sentence_vectors_np = np.array(sentence_vectors)


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Assuming sentence_vectors_np is your array of sentence vectors
tsne_model = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200)
reduced_vectors = tsne_model.fit_transform(sentence_vectors_np)

# Plotting
plt.figure(figsize=(16, 16))  # Increase figure size
for i, vec in enumerate(reduced_vectors):
    plt.scatter(vec[0], vec[1])
    plt.text(vec[0]+0.01, vec[1]+0.01, i, fontsize=14)  # Adjust text position

plt.title("Sentence Embeddings Visualized with t-SNE")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")


end_time = time.time()
elapsed_time = end_time - start_time
print(f"The code took {elapsed_time} seconds to execute.")


plt.show()