
import sqlite3
import pandas as pd
from Repertorio_Funciones import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("Data_Dia1.csv", sep = ",")

## Unicamente para facilitar la visualizacion
connection = sqlite3.connect(":memory:")
df.to_sql("tabla", connection, index = False, if_exists = "replace")
print_tabla(connection, "SELECT * FROM tabla")
connection.close()

# Definicion de X (Matriz/variables de decision) y de y (vector/variable a predecir)
# iloc se usa para poder tratar el df como un array multidimensional, y values devuelve
# los valores retirando los indices
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Como hay valores nan, usamos la funcion Imputer, que sustituye en
# dichos valores por una estadística del  conjunto, en este caso la media
imp = SimpleImputer(missing_values = np.nan, strategy = "mean")
imp = imp.fit(X[:, 1:])
X[:, 1:] = imp.transform(X[:, 1:])

# Ahora convertimos las columnas "Country" y "Purchased", en una clasificacion numerica
le_X = LabelEncoder()
le_y = LabelEncoder()
le_X.fit(X[:, 0])
le_y.fit(y)
X[:, 0] = le_X.transform(X[:, 0])
y = le_y.transform(y)

# Entrenamos el modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1, shuffle = True, stratify = y)


# Normalizamos los valores de las matrices, pues si no las distancias entre la tercera
# columna son mayores que entre la segunda, lo que genera un desbalance de los pesos asociados
sc = StandardScaler()
print(X_train,X_test)

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

print(X_train, X_test)

# Poco sentido con lo pequeña que es la base de datos, pero al menos practico la implementacion de decision trees
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


