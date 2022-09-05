import pandas as pd
import procesa_datos as d
import torch
import matplotlib.pyplot as plt
from Modelos import ModeloSimple
import ops
from tqdm import tqdm

"""
Visualizacion de datos: 
    - Cada una de los caracteres es un input (59 inputs??)
    - Cada input tiene 4 posiciones: A C T G
    - Hay 3 clases de salida: EI IE N
    - Habria que consultar el script de este señor de las casas a ver que hacia
"""

# Leemos los datos
csv = pd.read_csv("splice.data")

# Procesamos los datos
labels, data = d.procesa_datos(csv)
# Esto lo transforma en un formato guay para pytorch
train, val, test = d.to_pytorch(labels, data)

red = ModeloSimple()  # Declaramos nuestro modelo 
optim = torch.optim.SGD(red.parameters(), lr=0.05)  # Utilizamos SDG como optimizador

# Metricas de perdida y precision
tot_acc = []
tot_loss = []

# Entrenamiento y validacion en 100 epochs
for _ in tqdm(range(100)):

    ops.entrenamiento(train, red, optim)
    acc, loss, _ = ops.val(val, red, optim)
    tot_acc.append(acc)
    tot_loss.append(loss.item())

# Representacion de las precisiones over epochs
plt.plot(tot_acc)
plt.ylabel("Precision")
plt.xlabel("Iteraciones")
plt.title("Modelo Lineal FC")
plt.show()

# Test de la red
acc, _, acc_clases = ops.val(test, red, optim, test=True)

# Representacion de las precisiones finales de cada clase y la global
acc_clases.append(acc)
plt.bar(["EI", "IE", "N", "Global"],acc_clases)
plt.ylabel("Precision")
plt.xlabel("Clase")
plt.title("Precision final del modelo")
plt.show()