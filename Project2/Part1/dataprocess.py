import re
import os
import sys
import numpy as np
import logging


def build_vocab(file_paths):
    vocab = {"UNK": 0}
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    word, _ = line.split()  # 分割每行的单词和标签
                    if word not in vocab:
                        vocab[word] = len(vocab)
    return vocab


def build_tag2idx(filename):
    tags = []
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        tags = re.findall(r"[B|M|E|S|I]-[A-Z]+", content)
    if filename == "../NER/Chinese/tag.txt":
        tags = tags[4:]
    else:
        tags = tags[2:]
    tags.append("O")
    tag2idx = {}
    tag2idx["O"] = 0
    count = 1
    for tag in tags[:-1]:
        tag2idx[tag] = count
        count += 1
    return tag2idx


def read_data(file_path):
    sentences = []
    labels = []
    curr_sentence = []
    curr_labels = []

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
    valid_sentences, valid_labels = load_from_npz(valid_file)
    if mode == "test":
        test_sentences, test_labels = load_from_npz(test_file)

    logging.info(f"train dataset size: {len(train_sentences)}")
    logging.info(f"valid dataset size: {len(valid_sentences)}")
    if mode == "test":
        logging.info(f"test dataset size: {len(test_sentences)}")

    train_data = list(zip(train_sentences, train_labels))
    valid_data = list(zip(valid_sentences, valid_labels))
    if mode == "test":
        test_data = list(zip(test_sentences, test_labels))
    else:
        test_data = None
    return train_data, valid_data, test_data


def set_log(file_path):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_handlers = [logging.StreamHandler(stream=sys.stdout)]
    if file_path is not None:
        log_handlers.append(logging.FileHandler(file_path))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=log_handlers,
    )


def combine_data(sentences, labels):
    combined_data = []
    for sentence, label in zip(sentences, labels):
        combined_tokens = [f"{token} {tag}\n" for token, tag in zip(sentence, label)]
        combined_data.append("".join(combined_tokens))
    combined_data = "\n".join(combined_data)
    combined_data += "\n"
    return combined_data
