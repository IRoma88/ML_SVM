# Reconocimiento de Caras usando Eigenfaces y SVMs

Este proyecto implementa un sistema de reconocimiento facial utilizando el dataset preprocesado **Labeled Faces in the Wild (LFW)** y técnicas de reducción de dimensionalidad (PCA) junto con clasificadores SVM.

## 📁 Dataset

El dataset se descarga automáticamente usando `fetch_lfw_people` de `scikit-learn`. También puedes obtenerlo manualmente:

- http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz

Se utiliza una versión preprocesada del dataset con un mínimo de 70 imágenes por persona.

## 🚀 Descripción del proyecto

1. **Carga del dataset**
2. **Estandarización de imágenes**
3. **Reducción de dimensionalidad con PCA (eigenfaces)**
4. **Entrenamiento de un clasificador SVM con búsqueda aleatoria de hiperparámetros**
5. **Evaluación cualitativa y cuantitativa**
6. **Visualización de eigenfaces y predicciones**

## 🧪 Requisitos

Este proyecto requiere Python 3.8 o superior y las siguientes librerías (ver `requirements.txt`).

## 📦 Instalación

```bash
git clone https://github.com/tuusuario/reconocimiento-caras-eigenfaces.git
cd reconocimiento-caras-eigenfaces
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
````

## 📊 Resultados
Se visualizan:

  . Matriz de confusión normalizada

  . Imágenes de prueba con predicciones

  . Eigenfaces (componentes principales)

🔬 Mejoras futuras
En futuras versiones se puede:

  . Usar CNNs con TensorFlow/Keras

  . Implementar métricas de comparación con embeddings

  . Usar otros datasets de mayor resolución

## 📄 Licencia
Este proyecto es de uso académico y educativo. Puedes reutilizarlo con atribución.
