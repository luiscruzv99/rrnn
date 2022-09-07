import pandas as pd
import procesa_datos as d
import torch
import matplotlib.pyplot as plt
from Modelos import ModeloSimple, ModeloConv, Lineal
import torch.optim.lr_scheduler as o
import ops
from tqdm import tqdm


def vis_datos(data, labels):
    # Conteo de cada variable en la cadena
    # Ver la clase a la que pertenece

    ei = [[], [], [], []]
    ei_lab = [[],[],[],[]]
    for l in range(len(labels)):
        if labels[l] == [1, 0, 0]:
            a = 0
            c = 0
            t = 0
            g = 0

            for d in data[l][0]:
                if d == 0:
                    a += 1
                elif d == 1:
                    c += 1
                elif d == 2:
                    t += 1
                elif d == 3:
                    g += 1
            ei[0].append(a)
            ei_lab[0].append(1)
            ei[1].append(c)
            ei_lab[1].append(2)
            ei[2].append(t)
            ei_lab[2].append(3)
            ei[3].append(g)
            ei_lab[3].append(4)

    ie = [[], [], [], []]
    ie_lab = [[], [], [], []]
    for l in range(len(labels)):
        if labels[l] == [0, 1, 0]:
            a = 0
            c = 0
            t = 0
            g = 0

            for d in data[l][0]:
                if d == 0:
                    a += 1
                elif d == 1:
                    c += 1
                elif d == 2:
                    t += 1
                elif d == 3:
                    g += 1
            ie[0].append(a)
            ie_lab[0].append(1)
            ie[1].append(c)
            ie_lab[1].append(2)
            ie[2].append(t)
            ie_lab[2].append(3)
            ie[3].append(g)
            ie_lab[3].append(4)

    n = [[], [], [], []]
    n_lab = [[], [], [], []]
    for l in range(len(labels)):
        if labels[l] == [0, 1, 0]:
            a = 0
            c = 0
            t = 0
            g = 0

            for d in data[l][0]:
                if d == 0:
                    a += 1
                elif d == 1:
                    c += 1
                elif d == 2:
                    t += 1
                elif d == 3:
                    g += 1
            n[0].append(a)
            n_lab[0].append(1)
            n[1].append(c)
            n_lab[1].append(2)
            n[2].append(t)
            n_lab[2].append(3)
            n[3].append(g)
            n_lab[3].append(4)


    x = "Nucleótidos"
    y = "Apariciones por cadena"

    fig, axs = plt.subplots(3,1)
    axs[0].scatter(ei_lab,ei)
    axs[0].set_title("Apariciones nucleótidos en Exón-Intrón")
    axs[1].scatter(ie_lab,ie)
    axs[1].set_title("Apariciones nucleótidos en Intrón-Exón")
    axs[2].scatter(n_lab,n)
    axs[2].set_title("Apariciones nucleótidos en Ninguno")

    for ax in axs.flat:
        ax.set(xlabel=x, ylabel=y)
        ax.label_outer()

    plt.show()

def carga_datos(bs=32,  vis_datos=False):
    # Leemos los datos
    csv = pd.read_csv("splice.data")
    # Procesamos los datos
    labels, data = d.procesa_datos(csv)

    if vis_datos:
        vis_datos(data, labels)
    # Esto lo transforma en un formato guay para pytorch
    train, val, test = d.to_pytorch(labels, data, bs=bs)

    return train, val, test


def modelo_lineal(train, val, test):
    red = Lineal()
    optim = torch.optim.SGD(red.parameters(), lr=0.005)
    loss_fn = torch.nn.MSELoss()

    scheduler = o.StepLR(optim, step_size=10, gamma=0.5)

    tot_acc = []
    tot_loss = []
    t = tqdm(range(100))

    for i in t:
        ops.entrenamiento(train, red, optim, scheduler, loss_fn=loss_fn)
        acc, loss, _ = ops.val(val, red, optim, loss_fn=loss_fn)
        tot_acc.append(acc)
        tot_loss.append(loss.item())
        t.set_postfix({"MSELoss": loss.item(), "Acc": acc})

    dibuja_plot(tot_acc, "Lineal")

    acc, _, class_acc = ops.val(test, red, optim, loss_fn=loss_fn, test=True)
    class_acc.append(acc)
    dibuja_barra(class_acc, "Lineal")


def modelo_simple(train, val, test):
    red = ModeloSimple()
    optim = torch.optim.Adam(red.parameters(), lr=0.005)
    loss_fn = torch.nn.CrossEntropyLoss()

    scheduler = o.StepLR(optim, step_size=10, gamma=0.5)

    tot_acc = []
    tot_loss = []
    t = tqdm(range(100))

    for i in t:
        ops.entrenamiento(train, red, optim, scheduler, loss_fn=loss_fn)
        acc, loss, _ = ops.val(val, red, optim, loss_fn=loss_fn)
        tot_acc.append(acc)
        tot_loss.append(loss.item())
        t.set_postfix({"CELoss": loss.item(), "Acc": acc})

    dibuja_plot(tot_acc, "Simple FC")

    acc, _, class_acc = ops.val(test, red, optim, loss_fn=loss_fn, test=True)
    class_acc.append(acc)
    dibuja_barra(class_acc, "Simple FC")


def modelo_conv(train, val, test):
    red = ModeloConv()
    optim = torch.optim.Adam(red.parameters(), lr=0.005)
    loss_fn = torch.nn.CrossEntropyLoss()

    scheduler = o.StepLR(optim, step_size=40, gamma=0.4)

    tot_acc = []
    tot_loss = []
    class_acc = []

    t = tqdm(range(100))

    for i in t:
        ops.entrenamiento(train, red, optim, scheduler, loss_fn=loss_fn)
        acc, loss, _ = ops.val(val, red, optim, loss_fn=loss_fn)
        tot_acc.append(acc)
        tot_loss.append(loss.item())
        t.set_postfix({"CELoss": loss.item(), "Acc": acc})

    dibuja_plot(tot_acc, "Convolutivo")

    acc, _, class_acc = ops.val(test, red, optim, loss_fn=loss_fn, test=True)
    class_acc.append(acc)
    dibuja_barra(class_acc, "Convolutivo")


def dibuja_plot(accs, nombre):
    plt.plot(accs)
    plt.ylabel("Precision")
    plt.xlabel("Iteraciones")
    plt.title("Modelo "+nombre)
    plt.show()


def dibuja_barra(accs, nombre):
    plt.bar(["EI", "IE", "N", "Global"], accs)
    plt.ylabel("Precision")
    plt.xlabel("Clase")
    plt.title("Modelo "+nombre)
    plt.show()


if __name__ == "__main__":

    print("Modelo Lineal")
    train, val, test = carga_datos(bs=128, vis_datos=True)
    modelo_lineal(train, val, test)

    print("Modelo Simple FC")
    train, val, test = carga_datos(bs=128)
    modelo_simple(train, val, test)

    print("Modelo Convolutivo")
    train, val, test = carga_datos(bs=128)
    modelo_conv(train, val, test)
