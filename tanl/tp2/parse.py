import string

def preprocess_sentence(sentence):
    """
    Preprocess a single sentence: convert to lowercase and remove punctuation.
    """
    # Convert to lowercase
    sentence = sentence.lower()
    # Remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence

def parse_input_file(file_path):
    """
    Parse the input file to extract and preprocess sentences.
    Returns a list of tuples (sentence_id, preprocessed_sentence).
    """
    processed_sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split line into ID and sentence
            parts = line.strip().split('\t')
            if len(parts) == 2:
                sentence_id, sentence = parts
                # Preprocess the sentence
                preprocessed_sentence = preprocess_sentence(sentence)
                processed_sentences.append((sentence_id, preprocessed_sentence))

    return processed_sentences

# Replace with the path to your input file
input_file_path = 'T_sent.txt'
processed_data = parse_input_file(input_file_path)

# Optionally, print some processed sentences to verify
for i in range(5):
    print(processed_data[i])
