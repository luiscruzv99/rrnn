import pandas as pd

test = pd.read_csv("splice.data")

print(test)

labels_raw = []
data_raw = []

for row in test.iterrows():
    labels_raw.append(row[1][0].strip())
    tmp = row[1][2].strip()
    data_raw.append([[*tmp[:30]], [*tmp[30:]]]) # TODO: la cadena es de 59 chars, no 60, duplicado el caracter de en medio
    
print(labels_raw)
print(data_raw[0][0])
print(data_raw[0][1])
