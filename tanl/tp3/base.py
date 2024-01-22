def read_iob2_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    spacy_data, crfsuite_data = [], []
    current_sentence_spacy, current_sentence_crfsuite = [], []

    for line in lines:
        line = line.strip()
        if line:
            parts = line.split()  # Split the line into parts
            token, tag = parts[0], parts[-1]  # First element is the token, last is the tag
            current_sentence_spacy.append(token)
            current_sentence_crfsuite.append((token, tag))
        else:
            # Add the sentence to spaCy data as a single string
            if current_sentence_spacy:
                spacy_data.append(' '.join(current_sentence_spacy))
                crfsuite_data.append(current_sentence_crfsuite)
                current_sentence_spacy, current_sentence_crfsuite = [], []

    return spacy_data, crfsuite_data


# Example usage (replace 'path/to/file.iob2' with the actual file path)
spacy_sentences, crfsuite_sentences = read_iob2_file('data/UNER_Portuguese-Bosque-master/pt_bosque-ud-train.iob2')
