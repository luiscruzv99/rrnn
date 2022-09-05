import torch as pt
import torch.nn as nn
import torch.nn.functional as F

class ModeloSimple(nn.Module):
    def __init__(self):
        super(ModeloSimple, self).__init__()
        self.entrada = nn.Linear(60, 10)
        self.act_entrada = nn.ReLU()
        self.salida = nn.Linear(10, 3)
        self.act_salida = nn.Softmax()

    def forward(self, x):
        x = self.entrada(x)
        x = self.act_entrada(x)
        x = self.salida(x)
        x = self.act_salida(x)

        return x

class ModeloConv(nn.Module):
    # TODO: parametros de las capas
    def __init__(self):
        super(ModeloConv(), self).__init__()
        self.entrada = nn.Conv1D()
        self.act_entrada = nn.ReLU()
        self.pool = nn.MaxPool1d()

        self.intermedia = nn.Linear()
        self.act_intermedia = nn.ReLU()

        self.salida = nn.Linear()
        self.act_salida = nn.Softmax()

    def forward(self, x):
        x = self.entrada(x)
        x = self.act_entrada(x)
        x = self.pool(x)

        x = self.intermedia(x)
        x = self.act_intermedia(x)

        x = self.salida(x)
        x = self.act_salida(x)

        return x