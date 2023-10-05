import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from Adaline import Adaline
from sklearn.svm import SVC



# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=2,
                           random_state=9)
y[y == 0] = -1

# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)

# Entrenam un perceptron
perceptron = Adaline(eta=0.0005, n_iter=60)
perceptron.fit(X_transformed, y)
y_prediction = perceptron.predict(X)

#Entrenam una SVM linear (classe SVC)

# TODO
# Entrenar una SVM lineal
modelo_svm = SVC(C=1000,kernel='linear')
modelo_svm.fit(X_transformed, y)

plt.figure(1)

#  Mostram els resultats Adaline
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
m = -perceptron.w_[1] / perceptron.w_[2]
origen = (0, -perceptron.w_[0] / perceptron.w_[2])
plt.axline(xy1=origen, slope=m, c="blue", label="Adaline")


#  Mostram els resultats SVM
# TODO
#plt.axline(xy1= TODO, slope= TODO, c="green", label="SVM")
#plt.scatter( TODO, facecolors="none", edgecolors="green")

# Obtener la pendiente y el punto de intercepción con el eje y del hiperplano de la SVM
m_svm = -modelo_svm.coef_[0][0] / modelo_svm.coef_[0][1]
origen_svm = (0, -modelo_svm.intercept_[0] / modelo_svm.coef_[0][1])
plt.axline(xy1=origen_svm, slope=m_svm, c="green", label="SVM")
# Mostrar los puntos de las dos clases con diferentes colores
plt.scatter(X_transformed[y == -1][:, 0], X_transformed[y == -1][:, 1], c='red', marker='o', label='Clase -1')
plt.scatter(X_transformed[y == 1][:, 0], X_transformed[y == 1][:, 1], c='blue', marker='x', label='Clase 1')

plt.legend()
plt.show()
