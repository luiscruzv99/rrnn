import torch
import torch.nn as nn


def entrenamiento(loader, red, optim, sched, loss_fn=nn.MSELoss(reduction='mean')):


    for example, label in loader:
        optim.zero_grad()
        output = red.forward(example)
        loss = loss_fn(output, label)
        loss.backward()
        optim.step()
    sched.step()


def val(loader, red, optim, loss_fn=nn.MSELoss(reduction='mean'), test=False):
    red.eval()

    acc = 0.0
    loss = 0.0

    acc_class = [0, 0, 0]
    classes = [0, 0, 0]

    for example, label in loader:
        optim.zero_grad()
        output = red.forward(example)
        loss = loss_fn(output, label)

        guess = torch.max(output, -1)[1]
        true_val = torch.max(label, -1)[1]

        if test:
            classes[true_val.item()] += 1

        if guess.item() == true_val.item():
            if test:
                acc_class[guess.item()] += 1
            acc += 1

        loss += loss.item()

    acc = acc / len(loader)
    loss = loss / len(loader)
    if test:
        for i in range(len(classes)):
            try:
                acc_class[i] = acc_class[i] / classes[i]
            except ArithmeticError():
                acc_class[i] = 0
    return acc, loss, acc_class