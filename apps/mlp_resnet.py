import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
      nn.Residual(
        nn.Sequential(
          nn.Linear(dim, hidden_dim),
          norm(hidden_dim),
          nn.ReLU(),
          nn.Dropout(drop_prob),
          nn.Linear(hidden_dim, dim),
          norm(dim)
        )
      ),
      nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
      nn.Linear(dim, hidden_dim),
      nn.ReLU(),
      *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
      nn.Linear(hidden_dim, num_classes),
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    total_loss = 0
    total_errors = 0
    loss_fn = ndl.nn.SoftmaxLoss()
    if opt:
        model.train()
    else:
        model.eval()
    i = 0
    for inputs, labels in dataloader:
        # print(f"Step==============={i}, inputs.shape {inputs.shape}, labels {labels.shape}, model {model}")
        if opt:
            opt.reset_grad()
            y_prob = model(inputs)
            loss = loss_fn(y_prob, labels)
            loss.backward()
            opt.step()
        else:
            y_prob = model(inputs)
            loss = loss_fn(y_prob, labels)

        y_pred = np.argmax(y_prob.numpy(), axis=1)
        errors = np.not_equal(y_pred, labels.numpy()).sum()
        total_loss += loss.numpy() * inputs.shape[0]
        total_errors += errors
        i+=1
    total_examples = len(dataloader.dataset)
    return (total_errors/total_examples, total_loss/total_examples)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    resnet = MLPResNet(28*28, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = ndl.data.MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", 
                             f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = ndl.data.MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_loader = ndl.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_set, batch_size=batch_size)
    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, resnet, opt)
    test_err, test_loss = epoch(test_loader, resnet, None)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
