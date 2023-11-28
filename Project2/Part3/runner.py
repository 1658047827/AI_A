import torch
from sklearn import metrics
from tqdm import tqdm
from itertools import chain


class Runner:
    def __init__(self, model, optimizer, tag_num, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.tag_num = tag_num
        self.best_score = 0
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.model.to(self.device)

    def train(
        self,
        train_dataloader,
        valid_dataloader,
        epochs,
        device,
        save_path,
        eval_steps,
        eval_steps2,
        threshold,
    ):
        self.model.train()
        best_score = 0
        global_step = 0
        num_training_steps = epochs * len(train_dataloader)
        for epoch in range(1, epochs + 1):
            for text, label, seq_len in train_dataloader:
                self.model.state = "train"
                text = text.to(device)
                label = label.to(device)
                seq_len = seq_len.to(device)
                loss = self.model(text, seq_len, label)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(
                    f"epoch: [{epoch}/{epochs}], "
                    + f"loss: {loss.item():2.4f}, "
                    + f"step: [{global_step}/{num_training_steps}]"
                )
                global_step += 1

                if (
                    global_step % eval_steps == 0
                    or global_step == num_training_steps - 1
                ):
                    score = self.evaluate(valid_dataloader)
                    if score > threshold:
                        eval_steps = eval_steps2
                    self.model.train()
                    if score > best_score:
                        print(f"best score increase:{best_score} -> {score}")
                        best_score = score
                        self.save_model(save_path)

            print(f"training done best score: {best_score}")

    @torch.no_grad()
    def evaluate(self, valid_loader):
        self.model.eval()
        self.model.state = "eval"
        my_tags = []
        real_tags = []
        for sentence, valid_tags, sentence_len in tqdm(valid_loader):
            sentence = sentence.to(self.device)
            sentence_len = sentence_len.to(self.device)
            now_tags = self.model(sentence, sentence_len)
            for tags in now_tags:
                my_tags += tags
            for tags, now_len in zip(valid_tags, sentence_len):
                real_tags += tags[:now_len].tolist()
        score = metrics.f1_score(
            y_true=real_tags,
            y_pred=my_tags,
            labels=range(2, self.tag_num),
            average="micro",
        )
        return score

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        self.model.state = "pred"
        logits = self.model(x)
        return logits

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, model_path):
        model_state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_state_dict)
