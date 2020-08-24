# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:34:41 2020

@author: Ricardo
"""


import numpy as np
from sklearn  import datasets
import matplotlib.pyplot as plt

dataset = datasets.load_breast_cancer()
print(dataset)
print()

print('informacion del dataset')
print(dataset.keys())
print()

print('caracteristicas del dataset')
print(dataset.DESCR)
print()

print('cantidad de datos:')
print(dataset.data.shape)
print()

print('nombres de columnas:')
print(dataset.feature_names)
print()

X = dataset.data

y = dataset.target

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#se escalan los datos
from sklearn.preprocessing import StandardScaler

escalar = StandardScaler()

X_train = escalar.fit_transform(X_train)

X_test = escalar.transform(X_test)

from sklearn.linear_model import LogisticRegression

algoritmo = LogisticRegression()

#entrena el modelo

algoritmo.fit(X_train, y_train)

y_pred = algoritmo.predict(X_test)

from sklearn.metrics import confusion_matrix

matriz = confusion_matrix(y_test, y_pred)

print('matriz de confusion')

print(matriz)
print()

from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)

print('precision del modelo')
print(precision)
print()

#exactitud del modelo

from sklearn.metrics import accuracy_score

exactitud = accuracy_score(y_test, y_pred)

print('exactitud del modelo')

print(exactitud)
print()

#sensibilidad del modelo
from sklearn.metrics import recall_score

sensibilidad = recall_score(y_test, y_pred)

print('sensibilidad del modelo')
print(sensibilidad)
print()

from sklearn.metrics import f1_score

puntajef1 = f1_score(y_test, y_pred)
print('puntaje f1')
print(puntajef1)
print()

#Calculo curva ROC - AUC 

from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_pred)
print('Puntaje Curva Roc - Auc')
print(roc_auc)
