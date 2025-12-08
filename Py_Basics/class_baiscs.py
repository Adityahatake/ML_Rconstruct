class Model:
    def __init__(self, lr):
        self.lr = lr

    def train(self):
        print("Training with lr =", self.lr)

m = Model(0.01)
m.train()
