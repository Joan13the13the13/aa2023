import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from Adaline import Adaline
from sklearn.svm import SVC


# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

# Separar les dades: train_test_split

# TODO
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.7, train_size=0.3, random_state=None, shuffle=True, stratify=None)

# Estandaritzar les dades: StandardScaler

# TODO
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)

scalar_test = StandardScaler()
x_test = scalar_test.fit_transform(x_test)

# Entrenam una SVM linear (classe SVC)

# TODO
modelo = SVC(C = 10000, kernel = 'linear')
modelo.fit(x_train, y_train)

# Prediccio
# TODO
result = modelo.predict(x_test)
print(result)

# Metrica
# TODO
# Contar el número d'encerts (1)
num_aciertos = np.sum(result == 1)
# Mostrar el número de aciertos
print(f"Número de aciertos (1): {num_aciertos}")
# Calcular la precisión d'encerts
accuracy = accuracy_score(y_test, result)
# Mostrar la precisió
print(f"Precisión de aciertos: {accuracy * 100:.2f}%")
# Mostram la tasa d'encerts
tasa_encerts = num_aciertos/len(result)
print(f"La tasa d'encerts es "+ str(tasa_encerts))
