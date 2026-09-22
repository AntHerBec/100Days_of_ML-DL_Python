
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import make_column_transformer
from Repertorio_Funciones import *

df = pd.read_csv("50_Startups.csv", sep = ",")

connection = sqlite3.connect(":memory:")
df.to_sql("tabla", connection, index = False, if_exists = "replace")
print_tabla(connection, "SELECT * FROM tabla")
connection.close()

X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

# Como OneHotEncoder transforma en dummy variables todos los valores únicos del argumento, utilizo
# make_column_transformer para que esa transformación completa actúe únicamente sobre la columna
# -------------------------------------------------------------------------------------------------------
# make_column_transformer crea un ColumnTransformer que actua sobre 1 o varias columnas con las 1 o más
# transformaciones que especifico, y remainder = "passthrough" indica que no dropee el resto de columnas
ct = make_column_transformer((OneHotEncoder(categories = "auto"), [-1]), remainder = "passthrough")

X = ct.fit_transform(X)
print(X)

# Ya que las dummy variables se pueden expresar como 1 - (suma de las otras 2), la información que nos aporta
# una de ellas es redundante, y nos podemos deshacer de ella
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
columns = ["State = New York", "State = California", *df.columns]
result = f"Linear Regression: Profit = {lr.intercept_:.3f} +" if (lr.coef_[0] >= 0) else f"Linear Regression: Profit = {lr.intercept_:.3f} "
end = len(lr.coef_)-1
for i in range(end):
    if lr.coef_[i] >= 0:
        result += f"{lr.coef_[i]:3f}*({columns[i]}) + " if (lr.coef_[i+1] >= 0) else f"{lr.coef_[i]:3f}*({columns[i]}) "
    else:
        result += f"- {abs(lr.coef_[i]):3f}*({columns[i]}) + " if (lr.coef_[i+1] >= 0) else f"{lr.coef_[i]:3f}*({columns[i]}) "
result += f"{lr.coef_[end]:3f}*({columns[end]})" if (lr.coef_[end] >= 0) else f"- {abs(lr.coef_[end]):3f}*({columns[end]})"
print(result)

print(y_pred)

