class Optimizer:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.module_list:
            for key in layer.params.keys():
                layer.params[key] -= self.lr * layer.grads[key]

    def zero_grad(self):
        raise NotImplementedError