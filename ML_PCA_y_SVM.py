# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

"""# Reconocimiento de Caras usando eigenfaces y SVMs

El dataset usado en este proyecto es un dataset preprocesado obtenido desde "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

El objetivo de este proyecto es, dada una imagen de un rostro famoso debe clasificarse como tal de forma correcta.

Por ejemplo, si introducimos una imagen del presidente Bush al sistema, este debe ser capaz de arrojar el resultado correcto.
"""

from time import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from scipy.stats import loguniform

"""1. Obtenemos los datos, si no los tenemos en el disco, y los cargamos como arrays de numpy.


"""

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspección de las matrices de imágenes para encontrar las formas (para el trazado)
n_samples, h, w = lfw_people.images.shape

# para el aprendizaje automático utilizamos los 2 datos directamente
# (ya que la información de las posiciones relativas de los píxeles es ignorada por este modelo)
X = lfw_people.data
n_features = X.shape[1]

# la etiqueta a predecir es el id de la persona
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

"""2. Dividimos en training set y un test y mantenemos 25% el de los datos para el testing.

No olvides estandarizar los datos, ya que al ser imágenes es interesante que los valores sean entre 0 y 1 en lugar de entre 0 y 255.


"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""3. Calculamos el PCA (eigenfaces) en el dataset (tratado como un dataset sin etiquetar): reducción de la dimensionalidad o extracción de características no supevisado.


"""

n_components = 150
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
eightfaces = pca.components_.reshape((n_components, h, w))
print("done in %0.3fs" % (time() - t0))

pca_transform = pca.transform(X_train)
pca_test = pca.transform(X_test)

"""4. Entrenamos el modelo de clasificación SVM.


"""

param_distributions = {
    "C": loguniform(1e3, 1e5),
    "gamma": loguniform(1e-4, 1e-1),
}
clf = RandomizedSearchCV(SVC(kernel="rbf", class_weight="balanced"), param_distributions,  n_iter=10)

#clf = SVC(kernel='rbf', class_weight='balanced')
clf.fit(pca_transform, y_train)

print(clf.score(pca_test, y_test))
print(clf.best_estimator_)

"""5. Evaluamos el rendimiento del modelo en el conjunto de test.


"""

y_pred = clf.predict(pca_test)
print(classification_report(y_test, y_pred, target_names=target_names))
ConfusionMatrixDisplay.from_estimator(clf, pca_test, y_test, display_labels=target_names, cmap=plt.cm.Blues, normalize='true')
#plt.tight_layout()
plt.show()

"""6. Evaluamos el modelo sobre de forma cualitativa para ver si predice bien.

"""

plt.figure(figsize=(1.8 * 4, 2.4 * 5))
for i in range(4):
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_test[i].reshape(h, w), cmap=plt.cm.gray)
    plt.title(target_names[y_pred[i]].split()[-1],
              fontdict={'fontsize': 12})
    plt.xticks(())
    plt.yticks(())
plt.tight_layout()
plt.show()

"""Mostramos el resultado de la predicción en una parte de los datos de test.


"""

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits."""
    fig, axes = plt.subplots(n_row, n_col, figsize=(1.8 * n_col, 2.4 * n_row))
    fig.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)

    for ax, img, title in zip(axes.ravel(), images, titles):
        ax.imshow(img.reshape((h, w)), cmap='gray')
        ax.set_title(title, size=12)
        ax.set_xticks([])
        ax.set_yticks([])

def generate_titles(y_pred, y_test, target_names):
    """Generate prediction vs true labels for titles."""
    extract_name = lambda y: target_names[y].rsplit(" ", 1)[-1]
    return [f"predicted: {extract_name(y_pred[i])}\ntrue: {extract_name(y_test[i])}"
            for i in range(len(y_pred))]

# Generamos los títulos y graficamos en una sola línea
plot_gallery(X_test, generate_titles(y_pred, y_test, target_names), h, w)

"""7. Mostramos también ejemplos de los eigenfaces.


"""

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()

"""El problema del reconocimiento de caras se resolvería mucho más eficazmente entrenando redes neuronales convolucionales, pero esto lo daremos más adelante utilizando keras y
tensorflow para implementar dichos modelos.


"""
