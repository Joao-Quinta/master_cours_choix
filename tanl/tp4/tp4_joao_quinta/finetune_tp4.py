import os
from sklearn.metrics import classification_report
import time
from transformers import BertForTokenClassification
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch


class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)


def align_labels(tokenized_sentence, labels, label_map):
    word_ids = tokenized_sentence.word_ids()
    label_ids = []

    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)  # Special tokens
        elif word_idx != previous_word_idx:
            label_ids.append(label_map[labels[word_idx]])
        else:
            label_ids.append(label_map[labels[word_idx]])

        previous_word_idx = word_idx

    return label_ids


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
                token, tag = parts[1], parts[2]
                current_sentence.append(token)
                current_labels.append(tag)
        elif not line:
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)
                current_sentence, current_labels = [], []

    return sentences, labels


def train_model(train_loader, model, device, num_epochs=3, accumulation_steps=10):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps


def evaluate_model(model, dataloader, label_map):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    flat_predictions = []
    flat_true_labels = []
    label_map_inverse = {i: label for label, i in label_map.items()}
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
        for j in range(predictions.shape[0]):
            pred_labels = predictions[j].squeeze().tolist()
            true_labels_batch = labels[j].tolist()

            for pred, true_label in zip(pred_labels, true_labels_batch):
                if true_label != -100:
                    flat_predictions.append(label_map_inverse.get(pred, "UNKNOWN_LABEL"))
                    flat_true_labels.append(label_map_inverse.get(true_label, "UNKNOWN_LABEL"))
    return classification_report(flat_true_labels, flat_predictions, zero_division=0)


def prepare_data(sentences, label_lists, tokenizer, label_map, max_length_sent):
    max_length_sent = max_length_sent + 1
    tokenized_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
    aligned_labels_all = []
    for sentence, labels in zip(sentences, label_lists):
        if not sentence:
            continue
        tokenized_sentence = tokenizer(sentence, is_split_into_words=True,
                                       padding='max_length', truncation=True,
                                       max_length=max_length_sent, return_tensors="pt")
        for key in ['input_ids', 'attention_mask', 'token_type_ids']:
            tokenized_inputs[key].append(tokenized_sentence[key][0])
        aligned_labels = align_labels(tokenized_sentence, labels, label_map)
        aligned_labels += [-100] * (max_length_sent - len(aligned_labels))
        aligned_labels_all.append(aligned_labels)
    tokenized_inputs['labels'] = torch.tensor(aligned_labels_all, dtype=torch.long)
    return tokenized_inputs


def main():
    languages = ['Portuguese', 'Chinese', 'Swedish', 'Serbian', 'Slovak', 'Croatian', 'English', 'Danish']
    # languages = ['Portuguese']  # For testing with one language
    language_path = {
        'Portuguese': 'data/Portuguese/pt_bosque-ud-train.iob2',
        'English': 'data/English/en_ewt-ud-train.iob2',
        'Chinese': 'data/Chinese/zh_gsdsimp-ud-train.iob2',
        'Swedish': 'data/Swedish/sv_talbanken-ud-train.iob2',
        'Serbian': 'data/Serbian/sr_set-ud-train.iob2',
        'Croatian': 'data/Croatian/hr_set-ud-train.iob2',
        'Danish': 'data/Danish/da_ddt-ud-train.iob2',
        'Slovak': 'data/Slovak/sk_snk-ud-train.iob2'
    }

    labels_ = {
        'Portuguese': {'B-LOC': 0, 'I-LOC': 1, 'I-ORG': 2, 'B-ORG': 3, 'I-PER': 4, 'O': 5, 'B-PER': 6},
        'English': {'B-LOC': 0, 'I-LOC': 1, 'I-ORG': 2, 'B-ORG': 3, 'O': 4, 'I-PER': 5, 'B-PER': 6},
        'Chinese': {'B-LOC': 0, 'I-LOC': 1, 'I-ORG': 2, 'B-ORG': 3, 'O': 4, 'I-PER': 5, 'B-PER': 6},
        'Swedish': {'B-LOC': 0, 'I-LOC': 1, 'I-ORG': 2, 'B-ORG': 3, 'O': 4, 'I-PER': 5, 'B-PER': 6},
        'Serbian': {'B-LOC': 0, 'I-OTH': 1, 'I-LOC': 2, 'I-ORG': 3, 'B-ORG': 4, 'I-PER': 5, 'O': 6, 'B-OTH': 7,
                    'B-PER': 8},
        'Croatian': {'B-LOC': 0, 'I-OTH': 1, 'I-LOC': 2, 'I-ORG': 3, 'B-ORG': 4, 'I-PER': 5, 'O': 6, 'B-OTH': 7,
                     'B-PER': 8},
        'Danish': {'B-LOC': 0, 'I-LOC': 1, 'I-ORG': 2, 'B-ORG': 3, 'I-PER': 4, 'O': 5, 'B-PER': 6},
        'Slovak': {'B-LOC': 0, 'I-LOC': 1, 'I-ORG': 2, 'B-ORG': 3, 'I-PER': 4, 'O': 5, 'B-PER': 6}
    }
    # hot to use bert small model
    with open('compare_eval.txt', 'w') as file:
        for lan in languages:
            print(lan)
            model_name = "bert-base-multilingual-cased"
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
            unique_labels = set()
            sentences, label_lists = read_iob2_file(language_path[lan])

            sentences_test, label_lists_test = read_iob2_file(language_path[lan].replace("train", "test"))

            max_length = max(max([len(s) for s in sentences]), max([len(s) for s in sentences_test]))

            for labels in label_lists:
                unique_labels.update(labels)
            for labels in label_lists_test:
                unique_labels.update(labels)

            num_labels = len(unique_labels)
            model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            label_map = labels_[lan]
            print(label_map)
            tokenized_inputs = prepare_data(sentences, label_lists, tokenizer, label_map, max_length)
            tokenized_inputs_test = prepare_data(sentences_test, label_lists_test, tokenizer, label_map, max_length)

            model_path = f"./model2_train_{lan}"
            if not os.path.exists(model_path):
                train_dataset = NERDataset(tokenized_inputs, tokenized_inputs['labels'])
                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
                train_model(train_loader, model, device)
                # Optional: Save the fine-tuned model
                model.save_pretrained(model_path)

            model = BertForTokenClassification.from_pretrained(model_path)
            test_dataset = NERDataset(tokenized_inputs_test, tokenized_inputs_test['labels'])
            test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=4)
            file.write(lan)
            rep = evaluate_model(model, test_dataloader, label_map)
            file.write("\n")
            file.write(rep)
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

