
import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Repertorio_Funciones import *

df = pd.read_csv("Data_Dia4_1.csv", sep = ",")

# Vamos a retirar los puntos y coma de la última columna. para ello, utilizamos map,
# que sirve para alterarla empleando funciones o listados (listas, diccionarios...)
df.columns = ["grade1", "grade2", "label"]
df["label"] = df["label"].map(lambda x: float(x.rstrip(";")))  # Elimina el caracter ";" del string x

connection = sqlite3.connect(":memory:")
df.to_sql("tabla", connection, index = False, if_exists = "replace")
print_tabla(connection, "SELECT * FROM tabla")
connection.close()

# Debemos procesar los datos con simetría entre -1 y 1, debido a que logistic regression
# tiene como dominio los reales. Lo hacemos con MinMaxScaler
minmax = MinMaxScaler(feature_range = (-1,1))

X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
X = minmax.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Score by Logistic Regression in sklearn: ", logreg.score(X_test, y_test))  # Devuelve el accuracy score entre
                                                                                  # la muestra predicha y la real
admitted_index = np.where(y == 1.)
not_admitted_index = np.where(y == 0.)

# Función usada en la regresión logística,toma valores entre 0 y 1
def Sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def Sigmoid_surface(x1, x2, logreg = logreg):
    return Sigmoid(logreg.coef_[0][0]*x1 + logreg.coef_[0][1]*x2 + logreg.intercept_[0])


print(y_test)
print(logreg.predict(X_test))
print(logreg.coef_, logreg.intercept_)

# Definimos las matrices y variables que vamos a usar para la representación
axis_X = np.linspace(-1,1,101)
axis_Y = np.linspace(-1,1,101)
axis_X, axis_Y = np.meshgrid(axis_X, axis_Y)
print(axis_X)
Z = Sigmoid_surface(axis_X, axis_Y)

sigmoid_2d_index = np.where(abs(Z - 0.5) <= 1e-3)
print(sigmoid_2d_index)
Z_plano = np.full(Z.shape, 0.5)
z_admitted = Sigmoid_surface(X[admitted_index, 0], X[admitted_index, 1])
z_not_admitted = Sigmoid_surface(X[not_admitted_index, 0], X[not_admitted_index, 1])

# Visualizamos la distribución de puntos junto con la superficie dada por la función sigmoid en 2D y 3D
plt.scatter(X[admitted_index, 0], X[admitted_index, 1], c = "b")
plt.scatter(X[not_admitted_index, 0], X[not_admitted_index, 1], c = "r")
plt.plot(axis_X[sigmoid_2d_index], axis_Y[sigmoid_2d_index], c = "green")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
# La posición real de los puntos en el siguiente plot es y[index] pero para visualizar
# cómo se ajusta el threshold en 0.5 a los valores de y entendiendo la predicción es mejor así
ax.scatter(X[admitted_index, 0], X[admitted_index, 1], z_admitted, c = "b")
ax.scatter(X[not_admitted_index, 0], X[not_admitted_index, 1], z_not_admitted, c = "r")
ax.plot_surface(axis_X, axis_Y, Z, linewidth = 0, alpha = 0.2)
ax.plot_surface(axis_X, axis_Y, Z_plano, linewidth = 0, alpha = 0.2)
plt.show()









