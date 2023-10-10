class Classifier:
    def __init__(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError