import torch
from torch.utils.data import Dataset
from data_process import load_from_npz


class MyDataset(Dataset):
    def __init__(self, file_path, word2id, tag2id):
        self.file_path = file_path
        sentences, labels = load_from_npz(file_path)
        self.tag2id = tag2id
        self.word2id = word2id
        self.examples = []

        for sentence, label in zip(sentences, labels):
            t = [self.word2id.get(tk, self.word2id["UNK"]) for tk in sentence]
            l = [self.tag2id[lb] for lb in label]
            self.examples.append((t, l))

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

    def collect_fn(self, batch):
        text = [t for t, _ in batch]
        label = [l for _, l in batch]
        seq_len = [len(i) for i in text]
        max_len = max(seq_len)

        text = [t + [self.word2id["PAD"]] * (max_len - len(t)) for t in text]
        label = [l + [self.tag2id["O"]] * (max_len - len(l)) for l in label]

        text = torch.tensor(text, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        seq_len = torch.tensor(seq_len, dtype=torch.long)

        return text, label, seq_len
