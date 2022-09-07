import torch as pt
import torch.nn as nn
import torch.nn.functional as F

class Lineal(nn.Module):
    def __init__(self):
        super(Lineal, self).__init__()
        self.capa = nn.Linear(60, 3)

    def forward(self, x):
        return self.capa(pt.flatten(x, start_dim=1))


class ModeloSimple(nn.Module):
    def __init__(self):
        super(ModeloSimple, self).__init__()
        self.entrada = nn.Linear(60, 25)
        self.act_entrada = nn.ReLU()
        self.salida = nn.Linear(25, 3)
        self.act_salida = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.entrada(pt.flatten(x, start_dim=1))
        x = self.act_entrada(x)
        x = self.salida(x)
        x = self.act_salida(x)

        return x


class ModeloConv(nn.Module):
    def __init__(self):
        super(ModeloConv, self).__init__()
        self.entrada = nn.Conv1d(1, 1, 9, stride=1, padding=1)
        self.act_entrada = nn.ReLU()
        self.pool = nn.AvgPool1d(2)


        self.intermedia = nn.Linear(27, 15)
        self.drop= nn.Dropout()
        self.act_intermedia = nn.ReLU()

        self.salida = nn.Linear(15, 3)
        self.act_salida = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.entrada(x)
        x = self.act_entrada(x)
        x = self.pool(x)

        x = pt.flatten(x, start_dim=1)
        x = self.drop(x)
        x = self.intermedia(x)
        x = self.act_intermedia(x)

        x = self.salida(x)
        x = self.act_salida(x)

        return x
