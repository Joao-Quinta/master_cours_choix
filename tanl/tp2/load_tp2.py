from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string
from gensim.models import KeyedVectors
import math
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
    word_vectors = [model[word] for word in words if word in model.key_to_index]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)


def compute_similarity_matrix(sentence_vectors):
    return cosine_similarity(sentence_vectors)


def find_most_similar_sentences(processed_sentences, model):
    sentence_vectors = [sentence_to_vector(sentence, model) for _, sentence in processed_sentences]
    similarity_matrix = compute_similarity_matrix(sentence_vectors)
    np.fill_diagonal(similarity_matrix, -1) # to kill self similarity
    most_similar_sentences = {}
    for i, (sentence_id, _) in enumerate(processed_sentences):
        similar_sentence_idx = np.argmax(similarity_matrix[i])
        most_similar_sentence_id = processed_sentences[similar_sentence_idx][0]
        most_similar_sentences[sentence_id] = most_similar_sentence_id # set the value in a dictionary
    return most_similar_sentences


#################### END COMMON CODE FOR BOTH FILES


def load_google_news_model(local_model_path):
    model = KeyedVectors.load_word2vec_format(local_model_path, binary=True)
    return model


start_time = time.time()


input_file_path = 'T_sent.txt'
processed_data = parse_input_file(input_file_path)


# Example usage
local_model_path = 'GoogleNews-vectors-negative300.bin.gz'
model = load_google_news_model(local_model_path)
similar_sentences = find_most_similar_sentences(processed_data, model)

# Save results to out1.2.txt
with open('out1.2.txt', 'w', encoding='utf-8') as f:
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
