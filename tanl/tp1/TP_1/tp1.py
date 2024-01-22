# TANL TP1 JOAO QUINTA 25/10/2023
import string
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy.linalg import norm


#########################   write and read file   #########################


def read_from_file(filename):
    with open(filename, 'r') as f:
        return [word.strip() for word in f.readlines()]


def write_to_file(word_list, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        for word in word_list:
            file.write(word + '\n')


#########################   process   #########################


def preprocess(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        text = file.read()
        # Separate words by white spaces
        words = text.split()
        # Convert to lowercase and remove punctuation including double quotes
        punctuations = string.punctuation + '“”‘’'  # Adding typographic quotes
        table = str.maketrans("", "", punctuations)
        preprocessed_words = [word.lower().translate(table) for word in words]
    return preprocessed_words


# function that returns both B and T
# B is the 200 most frequent words
# T is the 100 most frequent words after B
# there are other possible ways of computing T, this one is a more general approach
def compute_B_and_T(words, B_size=200, T_size=100):
    # Count word frequencies
    word_freq = Counter(words)
    # Get the top B_size words for B
    B = [word for word, _ in word_freq.most_common(B_size)]
    # Get words after the top B_size words for T
    T = [word for word, _ in word_freq.most_common(B_size + T_size)][B_size:]
    return B, T


# compute co-occurrence matrix
# in short --> the function computes a matrix where each cell (i, j)
# contains the number of times word T[i] co-occurred with word B[j] within the specified window size
def compute_cooccurrence_matrix(words, T, B, window_size=5):
    # Initialize co-occurrence matrix with zeros
    cooccurrence_matrix = np.zeros((len(T), len(B)))
    half_window = window_size // 2
    for i in range(len(words)):
        if words[i] in T:
            # Get the window around the target word
            start_index = max(0, i - half_window)
            end_index = min(len(words), i + half_window + 1)
            # Get the context words in the window
            context = words[start_index:i] + words[i + 1:end_index]
            # Update the co-occurrence matrix
            for w in context:
                if w in B:
                    cooccurrence_matrix[T.index(words[i])][B.index(w)] += 1
    return cooccurrence_matrix


# compute ppmi matrix
# THIS is where we calculate and multiply by the feature weights
def compute_ppmi(matrix):
    # Compute the sums for each word and the total co-occurrences
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    total = np.sum(matrix)
    # Compute the expected co-occurrences if the words were independent
    expected = np.outer(row_sums, col_sums) / total
    # Compute the ratio
    ratio = matrix * total / (expected + 1e-8)  # deal with divide by 0
    # Apply the logarithm where the ratio is non-zero
    with np.errstate(divide='ignore'):
        ppmi_vals = np.log2(ratio)
    ppmi_vals[np.isnan(ppmi_vals)] = 0
    ppmi_vals[np.isinf(ppmi_vals)] = 0
    ppmi_vals = np.maximum(ppmi_vals, 0)  # Keep only the positive values
    return ppmi_vals


# computes and plots pca
def plot_pca(data, labels):
    # Apply PCA and transform the data to 2D
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)
    # Plot the transformed data
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    xs = pca_data[:, 0]  # first component
    ys = pca_data[:, 1]  # second component
    ax.scatter(xs, ys, s=50, alpha=0.6, edgecolors='w')
    # Add labels
    for x, y, label in zip(xs, ys, labels):
        ax.text(x, y, label)
    plt.show()


# compute cosine simularity
# THIS is where we calculate cosine_similarity
def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    magnitude = norm(A) * norm(B)
    if magnitude == 0:
        return 0
    return dot_product / magnitude


# compute most similar
def compute_most_similar(ppmi_matrix, T_words):
    num_words = len(T_words)
    similarity_matrix = np.zeros((num_words, num_words))
    for i in range(num_words):
        for j in range(num_words):
            similarity_matrix[i, j] = cosine_similarity(ppmi_matrix[i], ppmi_matrix[j])
    # For each word in T, find the most similar word
    most_similar = []
    for i in range(num_words):
        # The word shouldn't be compared with itself, so we set its similarity to a negative value
        similarity_matrix[i, i] = -1
        most_similar_idx = np.argmax(similarity_matrix[i])
        most_similar_word = T_words[most_similar_idx]
        most_similar.append((T_words[i], most_similar_word))
    return most_similar


#########################   input   #########################


print("Hi, I can compute B and T file for the dracula book (my raw text)")
print("Carry on or, use other raw_text, T and B files ?")
v = input("1 for costum input, anything else for default mode ---> ")
if v == "1":
    words_path = input("RAW TEXT PATH (include file type such as .txt : ")
    b_path = input("B PATH (include file type such as .txt : ")
    t_path = input("T PATH (include file type such as .txt : ")
    words = preprocess(words_path)
    B = read_from_file(b_path)
    T = read_from_file(t_path)
else:
    words = preprocess("raw_text.txt")
    B_w, T_w = compute_B_and_T(words)
    write_to_file(B_w, 'B.txt')
    write_to_file(T_w, 'T.txt')
    B = read_from_file('B.txt')
    T = read_from_file('T.txt')


co_matrix = compute_cooccurrence_matrix(words, T, B)
ppmi_matrix = compute_ppmi(co_matrix)
plot_pca(ppmi_matrix, T)
T_similarities = compute_most_similar(ppmi_matrix, T)

for i in T_similarities:
    print(i)
