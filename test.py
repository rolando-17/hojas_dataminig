import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn

data = pd.read_csv('datamining-python\hojas.csv')
clases = np.array(['Flor Morada','Flor Amarilla ','Flor Amarilla Delgada'])

print(data.head())
print(data.shape)

x=data[['largo', 'ancho']]
print(x.head())

y=data['clase']
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=0)

print("X_train shape: {}".format(X_train.values.shape))
print("y_train shape: {}".format(y_train.values.shape))
print("X_test shape: {}".format(X_test.values.shape))
print("y_test shape: {}".format(y_test.values.shape))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train.values,y_train.values)

X_new = np.array([[7.7,3.8]])
prediction = knn.predict(X_new)
print("Prediccion: {}".format(prediction))
print("Nombre del objetivo previsto: {}".format(clases[prediction]))
print("Puntaje del conjunto de prueba: {:.2f}".format(100*knn.score(X_test.values, y_test.values)))

dataframe = pd.DataFrame(X_train.values, columns=['largo', 'ancho'])
grr = pd.plotting.scatter_matrix(dataframe, c=y_train.values, figsize=(15,15),
                                marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8,cmap=mglearn.cm3)
plt.show();