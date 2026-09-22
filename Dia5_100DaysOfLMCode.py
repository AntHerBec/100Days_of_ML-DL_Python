
#########################################
### PREDICTION OF SURVIVAL IN TITANIC ###
#########################################

import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
from Repertorio_Funciones import *

#Visualización de los datos en tabla
df = pd.read_csv("titanic_train.csv", sep = ",")
connection = sqlite3.connect(":memory:")
df.to_sql("tabla", connection, index = False, if_exists = "replace")
print_tabla(connection, "SELECT * FROM tabla LIMIT 65")

# Visualización de los datos en plots

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 20))

axes = axes.flatten() # Transforma el array en 1D para que sea mas facil añadir los plots

# Visualización de los datos (uso de seaborn, módulo basado en matplotlib
# preparado para analizar de forma eficaz e informativa datos estadísticos.
# En inglés se puede llamar un EDA (Exploratory Data Analysis)
sns.countplot(x = "Survived", hue = "Survived", data = df, palette = "RdBu_r", legend = False, ax = axes[0])
axes[0].set_title("Survival Baseline Count")

sns.countplot(x = 'Survived', hue = 'Sex', data = df, palette ='RdBu_r', ax = axes[1])
axes[1].set_title("Survival Breakdown by Sex")

sns.countplot(x = 'Survived', hue = 'Pclass', data = df, palette = 'rainbow', ax = axes[2])
axes[2].set_title("Survival Breakdown by Passenger Class")

sns.histplot(df['Age'].dropna(), kde = False, color = 'darkred', bins=30, ax = axes[3])
axes[3].set_title("Age Distribution")

sns.countplot(x = 'SibSp', data = df, ax = axes[4])
axes[4].set_title("Count of Sibling/Spouses Aboard")

df['Fare'].hist(color = 'green', bins = 40, ax = axes[5])
axes[5].set_title("Fare Distribution")

sns.boxplot(x = 'Pclass', y = 'Age', hue = 'Pclass', data = df, palette = 'winter', legend = False,
            medianprops = {"color": "white"}, meanprops = {"color": "black"}, ax = axes[6])
axes[6].set_title("Age Distribution by Passenger Class")

axes[7].axis("off")  # Esconde el último bloque de forma limpia
plt.subplots_adjust(left=0.1, right=0.9, top=0.95,
                    bottom=0.05, hspace=0.4, wspace=0.3) # Usando tight_layout se superponen

plt.show()

# Vemos que no se cumplen condiciones importantes bajo análisis lineal de la variable
# dependiente, como homocestadicidad, distribución normal de los errores o pertenencia
# de la variable dependiente a un intervalo particular. Por otro lado sí se requiere
# independencia entre las observaciones y una muestra relativamente grande (usualmente
# se considera suficientemente grande si cada frecuencia en los valores de las variables
# independientes es igual o superior a 10, como 5 para aplicar chi cuadrado en inferencia)

# Para visualizar que clases tienen mas null, se puede hacer un mapa de calor como sigue
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
plt.show()
# Vemos que "cabin" es problemática al ser una variable no numérica, con muchos valores únicos
# y gran cantidad de valores nulos. Por tanto, no la consideraremos para el análisis. Por otro lado,
# los null en "Age" y "embarked" se tratarán de manera distinta:
#
# -"Age" es numérico y 'continuo', por lo que le asociamos la media por agrupación de sexo y pclass
# -"Embarked" es discreto y no numérico. Hay 3 opciones: asignar la moda (no hay tanta diferencia
#   entre las clases), eliminar las filas (perdemos la información) o hacemos una función que asigne
#   un valor aleatorio en función de las frecuencias de cada clase (no toma en cuenta lo bien que
#   pueda encajar la clase con los datos de la fila).
#
#   La última opción es seguramente la más adecuada para no perder datos en casos de muchos valores
#   nulos, pero como sólo hay 2 null es un trabajo innecesario ya que eliminar 2 filas no causará
#   un efecto significante ni problemas en el entrenamiento

### Tratamiento de nulls en "Embarked" (eliminar las filas) ###
null_index_Emb = df[df["Embarked"].isnull()].index
df = df.drop(null_index_Emb)

### Tratamiento de nulls en "Age" (media por pclass y sexo) ###
df["Age"] = df["Age"].fillna(
    df.groupby(["Pclass", "Sex"])["Age"].transform("mean")    # fillna completa los nan con
                                                              # las medias de las agrupaciones
)

columns = df.columns
X = df.iloc[:, [2, 4, 5, 6, 7, 9, 11]] # No usamos Id, survived (es la y), Name, Ticket ni Cabin
y = df.iloc[:, 1]

### Comprobación de correlación ###

# Buscamos correlacion entre las diferentes variables del dataframe utilizando la funcion corr().
# De no haber correlacion, sabemos que no habra (multi)colinearidad entre las variables

ct = make_column_transformer((OneHotEncoder(categories = "auto"), [1]), remainder = "passthrough")
ct2 = make_column_transformer((OneHotEncoder(categories = "auto"), [-1]), remainder = "passthrough")

# la primera columna representa si es hombre (1) o mujer (0), y podemos eliminarla
X = ct.fit_transform(X)[:, 1:]
# Lo mismo hacemos con el segundo transformer
X = ct2.fit_transform(X)[:, 1:]

correlation = pd.DataFrame(X, columns = ["Is_female", "Is_C_Emb", "Is_Q_emb", "Pclass", "Age",
                                         "SibSP", "Parch", "Fare"]).corr()

print(correlation.to_markdown())


# Los datos reflejan que ningun valor de correlacion supera (en valor absoluto) los 0.7-0.8, que se suele
# emplear como el límite a partir del cual la correlacion es suficientemente alta para confundir el modelo.
# Podemos entonces aplicar LogisticRegression sin preocuparnos en exceso
#
# Reescalamos los datos (Logistic Regression se apoya en algoritmos iterativos (lbfgs) para optimizar
# la función de costes, y escalas muy dispares pueden causar oscilaciones (por eso no converge)
sc = StandardScaler()
X = sc.fit_transform(X)

# Ahora entrenamos los datos como de costumbre y aplicamos la regresión
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0,
                                                    shuffle = True, stratify = y)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Imprimimos también la precisión de la predicción
print(accuracy_score(y_test, y_pred))

X_df = pd.DataFrame(X, columns = ["Is_female", "Is_C_Emb", "Is_Q_emb", "Pclass", "Age", "SibSP", "Parch", "Fare"])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
sns.regplot(x= 'Age', y= 'Survived', data= df, logistic= True, ax = axes[0]).set_title("Log Odds Linear Plot")
sns.regplot(x= 'Fare', y= 'Survived', data= df, logistic= True, ax = axes[1]).set_title("Log Odds Linear Plot")
plt.show()









