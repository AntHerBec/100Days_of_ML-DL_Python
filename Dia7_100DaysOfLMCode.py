
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, confusion_matrix
from Repertorio_Funciones import *

from sklearn.datasets import load_iris    # Dataset para la práctica de K-neighbors

iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
print(df.to_markdown())

X = iris.data[:,0:2]    # Datos de "sepal", "length" y "width" en cm. Usamos 2 para representación más clara
y = iris.target

sc = StandardScaler()
knn = KNeighborsClassifier(n_neighbors = 11, p = 2)   # p=2 implica la metrica de minkowski
                                                     # con potencia 2 (distancia euclidea)
X = sc.fit_transform(X)

# Creamos las muestras de entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size = 0.3,
                                                    shuffle = True, stratify = y)
neigh = knn.fit(X_train, y_train)
y_pred = neigh.predict(X_test)


# Representación en colores de los 3 tipos de flores (toda la muestra)
red, blue, yellow = [np.where(y == 0), np.where(y == 1), np.where(y == 2)]

fig, ax = plt.subplots(figsize=(8, 6))

# Podemos emplear DecisionBoundaryDisplay para mostrar como el algoritmo entrenado clasificaría
# los distintos puntos del grid. Es decir, separamos el plot en zonas divididas por clase. usando
# el atributo from_estimator, podemos tomar como guía el knn entrenado (se recomienda esta forma)
DecisionBoundaryDisplay.from_estimator(
    knn,
    X_test,
    response_method = "predict",
    plot_method = "pcolormesh",
    shading = "auto",
    alpha = 0.5,
    ax = ax
)
plt.scatter(X[red, 0], X[red, 1], c = "r", label = iris.target_names[0], edgecolors = "k")
plt.scatter(X[blue, 0], X[blue, 1], c = "b", label = iris.target_names[1], edgecolors = "k")
plt.scatter(X[yellow, 0], X[yellow, 1], c = "y", label = iris.target_names[2], edgecolors = "k")
plt.legend()
plt.title("Clasificación de tipos de flor" + "\n" + "(empleando k-nearest neighbors)")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.show()

# Representación en colores de los 3 tipos de flores (muestra de entrenamiento, de testeo y predichas)
fig = plt.figure(figsize = (10,8))
plot_grid = fig.add_gridspec(nrows = 2, ncols = 4)   # útil para representar varios plots de tamaño distinto
ax1 = fig.add_subplot(plot_grid[0,1:3])
ax2 = fig.add_subplot(plot_grid[1,:2])
ax3 = fig.add_subplot(plot_grid[1,2:])

axes = [ax1, ax2, ax3]
X_list = [X_train, X_test, X_test]
y_list = [y_train, y_test, y_pred]

for i in range(len(X_list)):

    red, blue, yellow = [np.where(y_list[i] == 0), np.where(y_list[i] == 1), np.where(y_list[i] == 2)]
    DecisionBoundaryDisplay.from_estimator(
        knn,
        X_test,
        response_method="predict",
        plot_method="pcolormesh",
        cmap = ListedColormap(["r", "b", "y"]),
        shading="auto",
        alpha=0.5,
        ax=axes[i]
    )
    axes[i].scatter(X_list[i][red, 0], X_list[i][red, 1], c="r", label=iris.target_names[0],
                    edgecolors="k")
    axes[i].scatter(X_list[i][blue, 0], X_list[i][blue, 1], c="b", label=iris.target_names[1],
                    edgecolors="k")
    axes[i].scatter(X_list[i][yellow, 0], X_list[i][yellow, 1], c="y", label=iris.target_names[2],
                    edgecolors="k")
    axes[i].legend()

    sample = "de entrenamiento"
    if i == 1: sample = "de prueba"
    if i == 2: sample = "predicha"

    axes[i].set_title("Clasificación de tipos de flor" + "\n" + f"(muestra {sample})")
    axes[i].set_xlabel(iris.feature_names[0])
    axes[i].set_ylabel(iris.feature_names[1])

plt.tight_layout()
plt.show()

