import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch


def procesa_datos(csv):
    """
    Procesado de los datos dado un csv en formato DataFrame de Pandas

    @type csv: Pandas.DataFrame
    @param csv: El .csv con datos en formato DataFrame de Pandas
    """

    labels_raw = []  # Etiquetas sin procesar
    data_raw = []  # Variables sin procesar
    
    labels = []  # Etiquetas procesadas
    data = []  # Datos procesados

    # "Barajeamos" el csv para que las entradas no esten ordenadas por clase
    csv = csv.sample(frac=1).reset_index(drop=True)

    # Recorremos todo el csv y partimos las filas en las columnas que necesitamos
    for row in csv.iterrows():
        labels_raw.append(row[1][0].strip())
        tmp = row[1][2].strip()
        data_raw.append([*tmp])  # Esto transforma un string en un array de caracteres
    

    # Procesado de las variables
    for d in data_raw:
        dn = []
        
        for e in d:
            if(e == "A"):
                dn.append(0)
            elif(e == "C"):
                dn.append(1)
            elif(e == "T"):
                dn.append(2)
            elif(e == "G"):
                dn.append(3)
            else:
                dn.append(4)

        data.append([dn])


    # Procesado de las etiquetas
    for l in labels_raw:
        if(l == "N"):
            labels.append([0,0,1])
        elif(l == "IE"):
            labels.append([0,1,0])
        elif(l == "EI"):
            labels.append([1,0,0])

    return labels, data


def to_pytorch(labels, data, split_train=0.8, split_test=0.9, bs=128):

    tam = len(labels)

    labels_train = torch.Tensor(labels[:int(tam*split_train)])
    data_train = torch.Tensor(data[:int(tam*split_train)])

    train_dataset = TensorDataset(data_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    labels_val = torch.Tensor(labels[int(tam*split_train):int(tam*split_test)])
    data_val = torch.Tensor(data[int(tam*split_train):int(tam*split_test)])

    val_dataset = TensorDataset(data_val, labels_val)
    val_loader = DataLoader(val_dataset, shuffle=True)

    labels_test = torch.Tensor(labels[int(tam*split_test):])
    data_test = torch.Tensor(data[int(tam*split_test):])

    test_dataset = TensorDataset(data_test, labels_test)
    test_loader = DataLoader(test_dataset, shuffle=True)

    return train_loader, val_loader, test_loader