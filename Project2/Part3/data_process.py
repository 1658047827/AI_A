import os
import re
import numpy as np

sorted_labels_eng = [
    "<PAD>",
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]

sorted_labels_chn = [
    "<PAD>",
    "O",
    "B-NAME",
    "M-NAME",
    "E-NAME",
    "S-NAME",
    "B-CONT",
    "M-CONT",
    "E-CONT",
    "S-CONT",
    "B-EDU",
    "M-EDU",
    "E-EDU",
    "S-EDU",
    "B-TITLE",
    "M-TITLE",
    "E-TITLE",
    "S-TITLE",
    "B-ORG",
    "M-ORG",
    "E-ORG",
    "S-ORG",
    "B-RACE",
    "M-RACE",
    "E-RACE",
    "S-RACE",
    "B-PRO",
    "M-PRO",
    "E-PRO",
    "S-PRO",
    "B-LOC",
    "M-LOC",
    "E-LOC",
    "S-LOC",
]


def read_data(file_path):
    sentences, labels = [], []
    curr_sentence, curr_labels = [], []

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line == "":
                if curr_sentence:
                    sentences.append(curr_sentence)
                    labels.append(curr_labels)
                    curr_sentence = []
                    curr_labels = []
                continue
            token, label = line.split(" ")
            curr_sentence.append(token)
            curr_labels.append(label)
    if len(curr_sentence) > 0:
        sentences.append(curr_sentence)
        labels.append(curr_labels)

    return sentences, labels


def save_as_npz(sentences, labels, file_path):
    array1 = np.array(sentences, dtype=object)
    array2 = np.array(labels, dtype=object)
    np.savez(file_path, sentences=array1, labels=array2)


def load_from_npz(file_path):
    loaded_data = np.load(file_path, allow_pickle=True)
    sentences = loaded_data["sentences"]
    labels = loaded_data["labels"]
    return sentences, labels


def data_process(folder_path, mode="train"):
    train_file = os.path.join(folder_path, "train.npz")
    valid_file = os.path.join(folder_path, "valid.npz")
    test_file = os.path.join(folder_path, "test.npz")

    if not os.path.isfile(train_file):
        train_path = os.path.join(folder_path, "train.txt")
        train_sentences, train_labels = read_data(train_path)
        save_as_npz(train_sentences, train_labels, train_file)

    if not os.path.isfile(valid_file):
        valid_path = os.path.join(folder_path, "validation.txt")
        valid_sentences, valid_labels = read_data(valid_path)
        save_as_npz(valid_sentences, valid_labels, valid_file)

    if mode == "test" and not os.path.isfile(test_file):
        test_path = os.path.join(folder_path, "test.txt")
        test_sentences, test_labels = read_data(test_path)
        save_as_npz(test_sentences, test_labels, test_file)

    train_sentences, train_labels = load_from_npz(train_file)
    train_data = list(zip(train_sentences, train_labels))

    valid_sentences, valid_labels = load_from_npz(valid_file)
    valid_data = list(zip(valid_sentences, valid_labels))

    if mode == "test":
        test_sentences, test_labels = load_from_npz(test_file)
        test_data = list(zip(test_sentences, test_labels))
    else:
        test_data = None

    return train_data, valid_data, test_data


def build_word2id(filename):
    word2id = {"PAD": 0, "UNK": 1}
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()
        if line:
            word, _ = line.split()
            if word not in word2id:
                word2id[word] = len(word2id)
    id2word = {v: k for k, v in word2id.items()}
    return word2id, id2word


def build_tag2id(filename):
    tags = []
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        tags = re.findall(r"[B|M|E|S|I]-[A-Z]+", content)
    if filename == "../NER/Chinese/tag.txt":
        tags = tags[4:]
    else:
        tags = tags[2:]
    tag2id = {}
    tag2id["O"] = 0
    for tag in tags:
        tag2id[tag] = len(tag2id)
    tag2id["<START>"] = len(tag2id)
    tag2id["<STOP>"] = len(tag2id)
    id2tag = {v: k for k, v in tag2id.items()}
    return tag2id, id2tag
