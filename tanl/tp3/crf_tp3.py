import pycrfsuite
from sklearn.metrics import classification_report
import time


def read_iob2_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    spacy_data, crfsuite_data = [], []
    current_sentence_spacy, current_sentence_crfsuite = [], []

    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):  # Skip comment lines
            parts = line.split('\t')  # Split the line into parts using tab
            if len(parts) >= 3:  # Check if there are enough parts (at least 3)
                token, tag = parts[1], parts[2]  # second element is the token, third is the tag
                current_sentence_spacy.append(token)
                current_sentence_crfsuite.append((token, tag))
        elif not line:
            # Add the sentence to spaCy data as a single string
            if current_sentence_spacy:
                spacy_data.append(' '.join(current_sentence_spacy))
                crfsuite_data.append(current_sentence_crfsuite)
                current_sentence_spacy, current_sentence_crfsuite = [], []

    return spacy_data, crfsuite_data


def predict_crfsuite(X_test, tagger):
    y_pred = []
    for xseq in X_test:
        y_pred.append(tagger.tag(xseq))
    return y_pred


def prepare_data_for_crfsuite(file_path):
    _, crfsuite_data = read_iob2_file(file_path)

    X = []
    y = []

    for sentence_labels in crfsuite_data:
        #print("--> ",sentence_labels)
        sentence_tokens = [token for token, _ in sentence_labels]
        sentence_label = [label for _, label in sentence_labels]
        X.append(sentence_tokens)
        y.append(sentence_label)
    return X, y




# Example usage


def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features





def main():
    languages = ['Portuguese', 'Chinese', 'Swedish', 'Serbian', 'Slovak', 'Croatian', 'English', 'Danish']
    #languages = ['Portuguese']


    language_path = {
        'Portuguese': 'data/Portuguese/pt_bosque-ud-test.iob2',
        'English': 'data/English/en_ewt-ud-test.iob2',
        'Chinese': 'data/Chinese/zh_gsdsimp-ud-test.iob2',
        'Swedish': 'data/Swedish/sv_talbanken-ud-test.iob2',
        'Serbian': 'data/Serbian/sr_set-ud-test.iob2',
        'Croatian': 'data/Croatian/hr_set-ud-test.iob2',
        'Danish': 'data/Danish/da_ddt-ud-test.iob2',
        'Slovak': 'data/Slovak/sk_snk-ud-test.iob2'
    }
    with open('crf_reports.txt', 'w') as file:
        for lang in languages:
            X_train, y_train = prepare_data_for_crfsuite(language_path[lang].replace('test','train'))
            X_test, y_test = prepare_data_for_crfsuite(language_path[lang])

            trainer = pycrfsuite.Trainer(verbose=False)

            for xseq, yseq in zip(X_train, y_train):
                if len(xseq) != len(yseq):
                    print("Mismatch found!")
                trainer.append(xseq, yseq)

            trainer.set_params({
                'c1': 1.0,  # L1 regularization
                'c2': 1e-3,  # L2 regularization
                'max_iterations': 50,
                'feature.possible_transitions': True
            })
            trainer.train('crf_model'+lang+'.crfsuite')

            # Predict
            tagger = pycrfsuite.Tagger()

            tagger.open('crf_model'+lang+'.crfsuite')
            y_pred = predict_crfsuite(X_test, tagger)
            flat_y_test = [label for seq in y_test for label in seq]
            flat_y_pred = [label for seq in y_pred for label in seq]

            # Print and Save the Classification Report
            report = classification_report(flat_y_test, flat_y_pred, zero_division=0)
            print(report)

            file.write(lang)
            file.write("\n")
            file.write(report)
            file.write("\n")
            file.write("\n")
            file.write("\n")
            file.write("\n")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Running time of the script: {total_time:.2f} seconds")