import spacy
from sklearn.metrics import classification_report
import os
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


def spacy_to_bio_format(doc):
    bio_tags = []
    i = 0
    while i < len(doc):
        token = doc[i]
        if token.ent_iob_ == 'O':  # Outside any named entity
            bio_tags.append('O')
            i += 1
        else:
            if token.ent_iob_ == 'B':  # Beginning of a named entity
                bio_tags.append(f'B-{token.ent_type_}')
                i += 1
            elif token.ent_iob_ == 'I':  # Inside a named entity
                bio_tags.append(f'I-{token.ent_type_}')
                i += 1
            else:
                raise ValueError(f'Invalid IOB tag: {token.ent_iob_}')

    return bio_tags


def evaluate_spacy_model(model, sentences, crfsuite_data):
    predictions = []
    true_labels = []

    for sentence, sentence_labels in zip(sentences, crfsuite_data):
        doc = model(sentence)
        preds = spacy_to_bio_format(doc)

        # Extract only the labels from the sentence_labels
        true_labels_sentence = [label for _, label in sentence_labels]

        if len(preds) == len(true_labels_sentence):
            predictions.extend(preds)
            true_labels.extend(true_labels_sentence)

    return classification_report(true_labels, predictions, zero_division=0)


def main():
    languages = ['Portuguese', 'Chinese', 'Swedish', 'Serbian', 'Slovak', 'Croatian', 'English', 'Danish']
    # languages = ['Portuguese']

    spacy_models = {
        'Portuguese': 'pt_core_news_md',
        'English': 'en_core_web_sm',
        'Chinese': 'zh_core_web_sm',
        'Swedish': 'sv_core_news_sm',
        'Serbian': 'xx_ent_wiki_sm',
        'Slovak': 'xx_ent_wiki_sm',
        'Croatian': 'hr_core_news_sm',
        'Danish': 'da_core_news_sm'
    }
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

    for lang in languages:
        # print(f"Evaluating {lang} language...")
        model = spacy.load(spacy_models[lang])
        test_file = language_path[lang]
        spacy_data, crfsuite_data = read_iob2_file(test_file)
        # print("len spacy data ---> ", len(spacy_data))
        # print(spacy_data)
        # print(crfsuite_data)
        evaluate_spacy_model(model, spacy_data, crfsuite_data)
def main():
    languages = ['Portuguese', 'Chinese', 'Swedish', 'Serbian', 'Slovak', 'Croatian', 'English', 'Danish']
    # languages = ['Portuguese']

    spacy_models = {
        'Portuguese': 'pt_core_news_md',
        'English': 'en_core_web_sm',
        'Chinese': 'zh_core_web_sm',
        'Swedish': 'sv_core_news_sm',
        'Serbian': 'xx_ent_wiki_sm',
        'Slovak': 'xx_ent_wiki_sm',
        'Croatian': 'hr_core_news_sm',
        'Danish': 'da_core_news_sm'
    }
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

    with open('spacy_rerports.txt', 'w') as file:
        for lang in languages:
            # print(f"Evaluating {lang} language...")
            model = spacy.load(spacy_models[lang])
            test_file = language_path[lang]
            spacy_data, crfsuite_data = read_iob2_file(test_file)
            # print("len spacy data ---> ", len(spacy_data))
            # print(spacy_data)
            # print(crfsuite_data)
            report = evaluate_spacy_model(model, spacy_data, crfsuite_data)
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
