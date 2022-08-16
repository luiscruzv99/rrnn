import pandas as pd

test = pd.read_csv("splice.data")

print(test)

labels_raw = []
data_raw = []

for row in test.iterrows():
    labels_raw.append(row[1][0].strip())
    tmp = row[1][2].strip()
    data_raw.append([[*tmp[:29]], [*tmp[30:]]]) # TODO: la cadena es de 59 chars, no 60, duplicado el caracter de en medio
    
print(labels_raw)
print(data_raw[0][0])
print(data_raw[0][1])

"""
Visualizacion de datos: 
    - Cada una de los caracteres es un input (59 inputs??)
    - Cada input tiene 4 posiciones: A C T G
    - Hay 3 clases de salida: EI IE N
    - Habria que consultar el script de este señor de las casas a ver que hacia
"""


"""
Red neuronal 1:
    - Una compuesta por 2 redes, una que coje una cadena de 30 chars y dice si es E o I
    - Se pasa 2 veces por ella, las 2 partes del genoma
    - Otra red que coje las 2 salidas y determina si es EI IE o N (esto puede ser una funcion logica)
TODO: Consultar los tipos de red que puede haber
"""
