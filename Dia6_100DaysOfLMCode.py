
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from Repertorio_Funciones import *

df = pd.read_csv("Social_Network_Ads.csv", sep = ",")
print(df.head(5).to_markdown()) # Algo más feo que el print_tabla, pero mucho más rápido de implementar

### Tratamiento de las variables en el dataframe ###

# En primer lugar comprobamos (visualmente) si hay valores nulos en el dataframe
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = "viridis")
plt.show()
# Vemos que no los hay. también se puede comprobar de manera más rigurosa como sigue
print(df.isnull().sum().sum())

# "Id" no es una variable que usaremos en el entrenamiento, por lo que debemos descartarla de las muestras
# para predecir. Además "Gender" es una variable cualitativa, que debemos convertir en numérica a través de
# un transformer. Nos quedaremos con una única columna, "is_male" o "is_female", ya que ambas se complementan
ct = make_column_transformer((OneHotEncoder(categories = "auto"), [0]), remainder = "passthrough")
X, y = df.iloc[:, 1:-1].values, df.iloc[:, -1].values
X = ct.fit_transform(X)
X = X[:, 1:]

# Como los valores de "Estimated Salary", "Age" y "Gender" son muy dispares, lo más recomendable es
# aplicar un reescalado para evitar oscilaciones al optimizar la función de costes (día 5). Usaremos
# el StandardScaler
ss = StandardScaler()
X_sc = ss.fit_transform(X)
print(X_sc)

# Creamos las muestras de entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, random_state = 0, test_size = 0.3,
                                                    shuffle = True, stratify = y)

# Llevamos a cabo la regresión logística, y calculamos la precisión de la predicción
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(accuracy_score(y_pred, y_test))

admited_index = np.where(y == 1.)
notadmitted_index = np.where(y == 0.)

# Otro enfoque sería, al momento de generar la muestra, utilizar 2 submuestras "male" y "female",
# entrenar los modelos con las variables "Age" y "Estimated Salary" y tener así un modelo logístico
# lineal para ambos casos, que también se pueden comparar entre sí
male_index = np.where(X[:, 0] == 1.)[0]
print(male_index)
fem_index = np.where(X[:, 0] == 0.)[0]

X_male = X[male_index, 1:]
X_fem = X[fem_index, 1:]
ss_male = StandardScaler()
ss_fem = StandardScaler()
X_male = ss_male.fit_transform(X_male)
X_fem = ss_fem.fit_transform(X_fem)
y_male = y[male_index]
y_fem = y[fem_index]

X_male_train, X_male_test, y_male_train, y_male_test = train_test_split(X_male, y_male, random_state = 0,
                                                    test_size = 0.3, shuffle = True, stratify = y_male)

X_fem_train, X_fem_test, y_fem_train, y_fem_test = train_test_split(X_fem, y_fem, random_state = 0,
                                                    test_size = 0.3, shuffle = True, stratify = y_fem)

print(X_male_train)
lr_male = LogisticRegression()
lr_fem = LogisticRegression()

lr_male.fit(X_male_train, y_male_train)
y_male_pred = lr_male.predict(X_male_test)
lr_fem.fit(X_fem_train, y_fem_train)
y_fem_pred = lr_fem.predict(X_fem_test)

accuracy_male = accuracy_score(y_male_pred, y_male_test)
accuracy_fem = accuracy_score(y_fem_pred, y_fem_test)
print(accuracy_male, accuracy_fem)

# Vemos que la separación en grupos, al estar entrenado con variables "continuas" o de muchas clases,
# proporciona  una mejora en la precisión respecto al caso original. No obstante, el modelo pierde
# generalidad. Podría representarse el primer caso en 3D y plantear el volumen correspondiente a la
# sigmoide (y la superficie correspondiente al threshold), así como obtener 2 representaciones 2D de
# los modelos específicos para ambos géneros.
#


def Sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def Sigmoid_surface(x1, x2, lr = LogisticRegression()):
    return Sigmoid(lr.coef_[0][0]*x1 + lr.coef_[0][1]*x2 + lr.intercept_[0])

axis_X = np.linspace(-10,10,1001)
axis_Y = np.linspace(-10,10,1001)
axis_X, axis_Y = np.meshgrid(axis_X, axis_Y)

Z_male = Sigmoid_surface(axis_X, axis_Y, lr = lr_male)
Z_fem = Sigmoid_surface(axis_X, axis_Y, lr = lr_fem)
sigmoid_2d_male = np.where(abs(Z_male - 0.5) <= 1e-3)
sigmoid_2d_fem = np.where(abs(Z_fem - 0.5) <= 1e-3)


fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (16,8))

admited_male, not_admited_male = np.where(y_male_train == 1.), np.where(y_male_train == 0.)
admited_fem, not_admited_fem = np.where(y_fem_train == 1.), np.where(y_fem_train == 0.)

axes[0,0].scatter(X_male_train[admited_male, 0], X_male_train[admited_male, 1], c = "green", edgecolors = "k")
axes[0,0].scatter(X_male_train[not_admited_male, 0], X_male_train[not_admited_male, 1], c = "r", edgecolors = "k")
axes[0,1].scatter(X_fem_train[admited_fem, 0], X_fem_train[admited_fem, 1], c = "green", edgecolors = "k")
axes[0,1].scatter(X_fem_train[not_admited_fem, 0], X_fem_train[not_admited_fem, 1], c = "r", edgecolors = "k")
axes[0,0].plot(axis_X[sigmoid_2d_male], axis_Y[sigmoid_2d_male])
axes[0,1].plot(axis_X[sigmoid_2d_fem], axis_Y[sigmoid_2d_fem])

admited_male, not_admited_male = np.where(y_male_test == 1.), np.where(y_male_test == 0.)
admited_fem, not_admited_fem = np.where(y_fem_test == 1.), np.where(y_fem_test == 0.)

axes[1,0].scatter(X_male_test[admited_male, 0], X_male_test[admited_male, 1], c = "green", edgecolors = "k")
axes[1,0].scatter(X_male_test[not_admited_male, 0], X_male_test[not_admited_male, 1], c = "r", edgecolors = "k")
axes[1,1].scatter(X_fem_test[admited_fem, 0], X_fem_test[admited_fem, 1], c = "green", edgecolors = "k")
axes[1,1].scatter(X_fem_test[not_admited_fem, 0], X_fem_test[not_admited_fem, 1], c = "r", edgecolors = "k")
axes[1,0].plot(axis_X[sigmoid_2d_male], axis_Y[sigmoid_2d_male])
axes[1,1].plot(axis_X[sigmoid_2d_fem], axis_Y[sigmoid_2d_fem])

j = 0

for i in np.ravel(axes):

    i.set_xlim(-3,3)
    i.set_ylim(-3,3)
    text = ["male train", "female train", "male test", "female test"]
    if j % 2 == 0:
        i.fill_between(axis_X[sigmoid_2d_male], axis_Y[sigmoid_2d_male], 3, color = "green", alpha = 0.3)
        i.fill_between(axis_X[sigmoid_2d_male], axis_Y[sigmoid_2d_male], -3, color = "r", alpha = 0.3)
    else:
        i.fill_between(axis_X[sigmoid_2d_fem], axis_Y[sigmoid_2d_fem], 3, color = "green", alpha = 0.3)
        i.fill_between(axis_X[sigmoid_2d_fem], axis_Y[sigmoid_2d_fem], -3, color = "r", alpha = 0.3)
    i.set_title("Classification for {} samples".format(text[j]))
    j += 1

plt.legend()
plt.show()

# Por último, en lugar de accuracy_score, vamos a evaluar la calidad de la predicción con confusion_matrix

fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (16,4))
print(y_test)
ConfusionMatrixDisplay.from_estimator(
    lr,
    X_test,
    y_test,
    display_labels = ["Not Purchased", "purchased"],
    ax = axes[0]) #con 3 variables
ConfusionMatrixDisplay.from_estimator(
    lr_male,
    X_male_test,
    y_male_test,
    display_labels = ["Not Purchased", "purchased"],
    ax = axes[1])
ConfusionMatrixDisplay.from_estimator(
    lr_fem,
    X_fem_test,
    y_fem_test,
    display_labels = ["Not Purchased", "purchased"],
    ax = axes[2])

titles = ["3 Variables", "2 Variables (male)", "2 Variables (female)"]
for i in range(len(titles)):
    axes[i].set_title(titles[i])
plt.tight_layout()
plt.show()






