import pycrfsuite
from sklearn.metrics import classification_report


def read_iob2_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    sentences, labels = [], []
    current_sentence, current_labels = [], []

    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split('\t')
            if len(parts) >= 3:
                token, label = parts[1], parts[2]
                current_sentence.append(token)
                current_labels.append(label)
        elif not line:
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)
                current_sentence, current_labels = [], []

    return sentences, labels


def word2features(sentence, i):
    word = sentence[i]
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
        word1 = sentence[i - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sentence) - 1:
        word1 = sentence[i + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features


def prepare_data_for_crfsuite(sentences, labels):
    X = [[word2features(s, i) for i in range(len(s))] for s in sentences]
    return X, labels


X_train, y_train = prepare_data_for_crfsuite(*read_iob2_file('data/Portuguese/pt_bosque-ud-train.iob2'))
X_test, y_test = prepare_data_for_crfsuite(*read_iob2_file('data/Portuguese/pt_bosque-ud-test.iob2'))

trainer = pycrfsuite.Trainer(verbose=False)
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,  # L1 regularization
    'c2': 1e-3,  # L2 regularization
    'max_iterations': 50,
    'feature.possible_transitions': True
})
trainer.train('crf_model.crfsuite')

tagger = pycrfsuite.Tagger()
tagger.open('crf_model.crfsuite')


def predict_crfsuite(X_test):
    y_pred = []
    for xseq in X_test:
        y_pred.append(tagger.tag(xseq))
    return y_pred


y_pred = predict_crfsuite(X_test)


def flatten(list_of_lists):
    """Flatten a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]


# Flatten the y_test and y_pred lists
y_test_flat = flatten(y_test)
y_pred_flat = flatten(y_pred)

# Now use these flat lists in the classification report
report = classification_report(y_test_flat, y_pred_flat)
print(report)

with open('crf_reports.txt', 'w') as file:
    file.write(report)
