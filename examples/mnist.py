"""Train a small MLP on MNIST using only Anvil(NumPy under the hood)"""

import time
import numpy as np
import anvil
from anvil import nn, optim
from anvil.data import DataLoader, load_mnist


anvil.manual_seed(0)

(x_train, y_train), (x_test, y_test)=load_mnist()


MEAN, STD=0.1307, 0.3081
x_train=((x_train - MEAN) / STD).reshape(-1, 784).astype(np.float32)
x_test=((x_test - MEAN) / STD).reshape(-1, 784).astype(np.float32)

model=nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10),
)
print(model)
print(f"parameters: {model.num_parameters():,}")

optimizer=optim.Adam(model.parameters(), lr=1e-3)
train_loader=DataLoader(x_train, y_train, batch_size=128, shuffle=True)
test_loader=DataLoader(x_test, y_test, batch_size=1000)


def evaluate():
    model.eval()
    correct=0
    with anvil.no_grad():
        for xb, yb in test_loader:
            correct+= (model(xb).data.argmax(axis=1)==yb).sum()
    return correct/len(x_test)



for epoch in range(1, 6):
    model.train()
    start, total_loss=time.time(), 0.0
    for xb, yb in train_loader:
        loss=nn.functional.cross_entropy(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"epoch {epoch}  loss {total_loss / len(train_loader):.4f}  "
          f"test acc {evaluate():.2%}  ({time.time() - start:.1f}s)")



model.save("mnist_mlp.npz")
print("saved weights to mnist_mlp.npz")
        