import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import logging
from torchmetrics import Accuracy


def set_device(gpu=-1):
    if gpu != -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    return device


class CNNClassifier:
    def __init__(self, model, gpu):
        self.model: torch.nn.Module = model
        self.best = {
            "train_loss": float("inf"),
            "valid_loss": float("inf"),
            "train_acc": 0.0,
            "valid_acc": 0.0,
        }
        self.device = set_device(gpu)
        self.model.to(self.device)

    def fit(self, train_loader, valid_loader, epoches, learning_rate, **kwargs):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = F.cross_entropy
        metric = Accuracy(task="multiclass", num_classes=12, top_k=1)
        metric.to(self.device)
        save_path = kwargs.get("save_path", "./CNN_best.ckpt")
        log_interval = kwargs.get("log_interval", 10)

        for epoch in range(1, epoches + 1):
            epoch_loss = 0.0
            metric.reset()
            epoch_time_start = time.time()
            for batch in train_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.long().to(self.device)
                preds = self.model(x)
                loss = loss_fn(preds, y)
                epoch_loss += loss.item()
                metric.update(preds, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss /= len(train_loader)
            epoch_acc = metric.compute()
            epoch_time_elapsed = time.time() - epoch_time_start
            if epoch % log_interval == 0:
                logging.info(
                    "Train epoch {}/{}, learning rate: {}, train loss: {:.6f}, train_acc: {:.6f} [{:.2f}s]".format(
                        epoch,
                        epoches,
                        optimizer.state_dict()["param_groups"][0]["lr"],
                        epoch_loss,
                        epoch_acc,
                        epoch_time_elapsed,
                    )
                )
            # evaluate
            result = self.evaluate(valid_loader, loss_fn, metric)
            if result["valid_acc"] > self.best["valid_acc"]:
                logging.info(
                    "Best validation accuracy has been updated: {:.6f} -> {:.6f}".format(
                        self.best["valid_acc"], result["valid_acc"]
                    )
                )
                result["train_loss"] = epoch_loss
                result["train_acc"] = epoch_acc
                self.best = result
                if save_path is not None:
                    self.save_model(save_path)
        
        self.load_model(save_path)

    @torch.no_grad()
    def evaluate(self, valid_loader, loss_fn, metric: Accuracy):
        self.model.eval()
        metric.reset()
        valid_loss = 0.0
        for batch in valid_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.long().to(self.device)
            preds = self.model(x)
            loss = loss_fn(preds, y)
            valid_loss += loss.item()
            metric.update(preds, y)
        valid_loss /= len(valid_loader)
        valid_acc = metric.compute()
        return {"valid_loss": valid_loss, "valid_acc": valid_acc}

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        preds = self.model(x)
        return preds

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ckpt = {"best": self.best, "state_dict": self.model.state_dict()}
        torch.save(ckpt, save_path)
        logging.info("Save model to {}".format(save_path))

    def load_model(self, load_path):
        os.makedirs(os.path.dirname(load_path), exist_ok=True)
        ckpt = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.best = ckpt["best"]
        logging.info("Load model from {}".format(load_path))
        logging.info("best pretrain: {}".format(self.best))
