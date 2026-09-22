
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from Repertorio_Funciones import *

# No es necesario la importación de LabelEncoder ni Imputer
# como en el dia 1 ya que no hay nan ni columnas no numericas
df = pd.read_csv("studentscores.csv", sep = ",")

connection = sqlite3.connect(":memory:")
df.to_sql("tabla", connection, index = False, if_exists = "replace")
print_tabla(connection, "SELECT * FROM tabla")
connection.close()

# Puesto que no tenemos que manipular los datos de la database
# asignamos directamente X a "Hours" e y a "Scores"
X, y = df.iloc[:,:-1].values, df.iloc[:,1].values

# Entrenamos el modelo sin usar stratify, puesto que y no está organizado por clases (como "Yes" o "No")
# sino que son valores numéricos varios. stratify sirve para que haya la misma proporcion de clases en la
# training sample que en la testing sample. Por lo mismo, no es necesario hacer un shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Tampoco es necesario un reescalado al tener una única variable, horas (además, al ser una variable "física"
# sería necesario reconvertirla a los valores originales para evitar datos ilícitos tiempos de estudio negativos)
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"Linear Regression: Score = {lr.intercept_:.3f} + {lr.coef_[0]:.3f} * (number of hours studying)")
y_pred = lr.predict(X_test)
print(y_pred)

plt.plot(X_train, lr.intercept_ + lr.coef_[0]*X_train, c = "orange", label = "Linear Regression")
plt.scatter(X_train, y_train, c = "red", label = "Training Data", zorder = 2)
plt.scatter(X_test, y_test, c = "blue", marker = "s", s = 40, label = "Testing Data", zorder = 2)
plt.scatter(X_test, y_pred, c = "black", marker = "+", s = 50, label = "Predicted Data",zorder = 2)
plt.legend()
plt.show()




