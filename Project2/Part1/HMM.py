import numpy as np
from tqdm import tqdm


class HMM_model:
    def __init__(self, tag2id, vocab=None):
        self.tag2id = tag2id
        self.n_tag = len(self.tag2id)  # 标签（隐状态）个数
        self.vocab = vocab
        self.epsilon = 1e-100
        self.idx2tag = {idx: tag for (tag, idx) in self.tag2id.items()}
        self.A = np.zeros((self.n_tag, self.n_tag))
        self.B = np.zeros((self.n_tag, len(vocab)))  # 第一列对应"UNK"
        self.Pi = np.zeros(self.n_tag)

    def train(self, train_data):
        for sentence, labels in tqdm(train_data):
            for j in range(len(sentence)):
                cur_word = sentence[j]
                cur_tag = labels[j]
                self.B[self.tag2id[cur_tag]][self.vocab[cur_word]] += 1
                if j == 0:  # 统计初始概率
                    self.Pi[self.tag2id[cur_tag]] += 1
                    continue
                pre_tag = labels[j - 1]
                self.A[self.tag2id[pre_tag]][self.tag2id[cur_tag]] += 1

        # 对数据进行对数归一化，并最后取log防止连乘浮点数下溢
        self.Pi = self.Pi / self.Pi.sum()
        self.Pi[self.Pi == 0.0] = self.epsilon
        self.Pi = np.log10(self.Pi)

        row_sums = self.A.sum(axis=1)
        zero_rows = row_sums == 0
        self.A[~zero_rows] /= row_sums[~zero_rows, None]
        self.A[zero_rows] = 0
        self.A[self.A == 0] = self.epsilon
        self.A = np.log10(self.A)

        row_sums = self.B.sum(axis=1)
        zero_rows = row_sums == 0
        self.B[~zero_rows] /= row_sums[~zero_rows, None]
        self.B[zero_rows] = 0
        self.B[self.B == 0] = self.epsilon
        self.B = np.log10(self.B)

        # 为"UNK"未知字符设置矩阵值（超参）
        self.B[:, 0] = np.ones(self.n_tag) * np.log(1.0 / self.n_tag)

    def valid(self, valid_data):
        return [self.viterbi(sentence) for sentence, _ in valid_data]

    def viterbi(self, O):
        N = len(self.Pi)
        T = len(O)
        delta = np.zeros((N, T))
        psi = np.zeros((N, T), dtype=int)

        delta[:, 0] = self.Pi + self.B[:, self.vocab.get(O[0], 0)]
        for t in range(1, T):
            O_t = self.vocab.get(O[t], 0)
            for j in range(N):
                # delta[j, t] = max(delta[i, t-1] * A[i, j] * B[j, O[t]]) 1<=i<=N
                # 参数已经取过log了，所以这里得用加法
                delta[j, t] = np.max(delta[:, t - 1] + self.A[:, j]) + self.B[j, O_t]
                psi[j, t] = np.argmax(delta[:, t - 1] + self.A[:, j])

        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(delta[:, -1])
        for t in range(T - 2, -1, -1):
            best_path[t] = psi[best_path[t + 1], t + 1]

        return [self.idx2tag[id] for id in best_path]

    def save_param(self, filename, format="txt"):
        if format == "txt":
            np.savetxt(f"{filename}_A.txt", self.A)
            np.savetxt(f"{filename}_B.txt", self.B)
            np.savetxt(f"{filename}_Pi.txt", self.Pi)
        elif format == "npy":
            np.save(f"{filename}_A.npy", self.A)
            np.save(f"{filename}_B.npy", self.B)
            np.save(f"{filename}_Pi.npy", self.Pi)
        elif format == "npz":
            np.savez(f"{filename}.npz", A=self.A, B=self.B, Pi=self.Pi)

    def load_param(self, filename, format="txt"):
        if format == "txt":
            self.A = np.loadtxt(f"{filename}_A.txt")
            self.B = np.loadtxt(f"{filename}_B.txt")
            self.Pi = np.loadtxt(f"{filename}_Pi.txt")
        elif format == "npy":
            self.A = np.load(f"{filename}_A.npy")
            self.B = np.load(f"{filename}_B.npy")
            self.Pi = np.load(f"{filename}_Pi.npy")
        elif format == "npz":
            loaded = np.load(f"{filename}.npz")
            self.A = loaded["A"]
            self.B = loaded["B"]
            self.Pi = loaded["Pi"]

    def predict(self, O):
        results = self.viterbi(O)
        for word, label in zip(O, results):
            print(f"{word} {label}")
