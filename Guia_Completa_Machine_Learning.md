# Guía Completa de Machine Learning: Modelos Supervisados y Evaluación

## Tabla de Contenidos
1. [¿Qué es el Machine Learning?](#qué-es-el-machine-learning)
2. [Modelos Supervisados](#modelos-supervisados)
3. [Evaluación de Modelos](#evaluación-de-modelos)
4. [Validación Cruzada](#validación-cruzada)
5. [GridSearchCV](#gridsearchcv)
6. [Benchmarking y Mejores Prácticas](#benchmarking-y-mejores-prácticas)

---

## ¿Qué es el Machine Learning?

Imagina que quieres enseñar a un niño a reconocer diferentes tipos de frutas. Le muestras muchas manzanas diciéndole "esto es una manzana", muchas naranjas diciéndole "esto es una naranja", etc. Después de ver muchos ejemplos, el niño aprende a identificar frutas nuevas que nunca había visto antes.

El **Machine Learning (Aprendizaje Automático)** funciona de manera similar: le mostramos a una computadora muchos ejemplos con sus respuestas correctas, y ella aprende patrones para hacer predicciones sobre datos nuevos.

### Tipos de Aprendizaje Automático

```python
# Ejemplo conceptual de los tipos de ML
"""
SUPERVISADO: Tenemos ejemplos con respuestas correctas
- Ejemplo: Fotos de gatos (X) + etiqueta "gato" (y)
- Objetivo: Predecir si una foto nueva es un gato

NO SUPERVISADO: Solo tenemos datos, sin respuestas
- Ejemplo: Solo fotos de animales (X)
- Objetivo: Agrupar animales similares

REFUERZO: Aprendemos mediante prueba y error
- Ejemplo: Un robot que aprende a caminar
- Objetivo: Maximizar recompensas (pasos dados sin caer)
"""
```

---

## Modelos Supervisados

### ¿Qué son los Modelos Supervisados?

Un modelo supervisado es como un estudiante que aprende con un profesor. El "profesor" son los datos de entrenamiento que incluyen tanto las preguntas (características) como las respuestas correctas (etiquetas).

**Ejemplo de la vida real:**
- **Predecir si un email es spam**: Le mostramos 10,000 emails etiquetados como "spam" o "no spam"
- **Predecir el precio de una casa**: Le mostramos 5,000 casas con sus precios de venta

### Tipos de Problemas Supervisados

#### 1. Clasificación
Predecir categorías o clases discretas.

```python
# Ejemplo: Clasificar emails como spam o no spam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Datos de ejemplo (en la realidad serían miles)
emails = [
    "¡GANA DINERO FÁCIL! Haz clic aquí",
    "Reunión mañana a las 10:00",
    "¡OFERTA LIMITADA! Compra ahora",
    "¿Vienes a cenar el viernes?",
    "Tu cuenta será suspendida, haz clic aquí",
    "Feliz cumpleaños, espero que lo pases genial"
]

# Etiquetas: 1 = spam, 0 = no spam
labels = [1, 0, 1, 0, 1, 0]

# Convertir texto a números (la computadora solo entiende números)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
y = np.array(labels)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo
model = MultinomialNB()
model.fit(X_train, y_train)

# Hacer predicciones
predictions = model.predict(X_test)

print("Precisión:", accuracy_score(y_test, predictions))

# Probar con un email nuevo
nuevo_email = ["¡Gana dinero desde casa!"]
nuevo_email_vector = vectorizer.transform(nuevo_email)
prediccion = model.predict(nuevo_email_vector)
print("¿Es spam?", "Sí" if prediccion[0] == 1 else "No")
```

#### 2. Regresión
Predecir valores numéricos continuos.

```python
# Ejemplo: Predecir el precio de una casa
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Datos de ejemplo: área de la casa (m²) vs precio (€)
areas = np.array([50, 80, 100, 120, 150, 200, 250, 300]).reshape(-1, 1)
precios = np.array([100000, 150000, 180000, 220000, 280000, 350000, 420000, 500000])

# Crear y entrenar el modelo
modelo_casa = LinearRegression()
modelo_casa.fit(areas, precios)

# Predecir el precio de una casa de 180 m²
area_nueva = np.array([[180]])
precio_predicho = modelo_casa.predict(area_nueva)
print(f"Una casa de 180 m² costaría aproximadamente: {precio_predicho[0]:,.0f}€")

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.scatter(areas, precios, color='blue', label='Datos reales')
plt.plot(areas, modelo_casa.predict(areas), color='red', label='Línea de predicción')
plt.xlabel('Área (m²)')
plt.ylabel('Precio (€)')
plt.title('Predicción de Precios de Casas')
plt.legend()
plt.show()
```

### Algoritmos de Modelos Supervisados

#### 1. Regresión Logística

**¿Qué es?**
A pesar del nombre, se usa para **clasificación**. Es como una "función matemática" que convierte cualquier número en un valor entre 0 y 1, que interpretamos como probabilidad.

**Ejemplo conceptual:**
Imagina que quieres predecir si un estudiante aprobará un examen basándose en las horas que estudió:
- 0 horas → 10% probabilidad de aprobar
- 5 horas → 50% probabilidad de aprobar
- 10 horas → 90% probabilidad de aprobar

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Crear datos de ejemplo
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                         n_informative=2, n_clusters_per_class=1, random_state=42)

# Entrenar modelo de regresión logística
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Hacer predicciones con probabilidades
probabilidades = log_reg.predict_proba(X)
print("Primeras 5 predicciones:")
for i in range(5):
    print(f"Ejemplo {i+1}: Clase {y[i]} - Probabilidades: {probabilidades[i]}")

# Visualizar la frontera de decisión
plt.figure(figsize=(12, 5))

# Gráfico 1: Datos originales
plt.subplot(1, 2, 1)
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', label='Clase 0')
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='s', label='Clase 1')
plt.title('Datos de Clasificación')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()

# Gráfico 2: Con frontera de decisión
plt.subplot(1, 2, 2)
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = log_reg.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', label='Clase 0')
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='s', label='Clase 1')
plt.title('Regresión Logística - Frontera de Decisión')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()

plt.tight_layout()
plt.show()
```

#### 2. Random Forest (Bosque Aleatorio)

**¿Qué es?**
Imagina que tienes una pregunta difícil y en lugar de preguntarle a una sola persona, le preguntas a 100 expertos diferentes y tomas la respuesta que más repiten. Random Forest hace exactamente eso, pero con "árboles de decisión".

**¿Qué es un árbol de decisión?**
Es como un diagrama de flujo de preguntas:
```
¿La persona tiene más de 30 años?
├── SÍ → ¿Gana más de 50k al año?
│   ├── SÍ → Probablemente comprará el producto
│   └── NO → Probablemente no comprará
└── NO → ¿Es estudiante universitario?
    ├── SÍ → Probablemente comprará (versión estudiante)
    └── NO → Probablemente no comprará
```

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd

# Cargar dataset famoso de flores Iris
iris = load_iris()
X, y = iris.data, iris.target

# Crear un DataFrame para mejor visualización
df = pd.DataFrame(X, columns=iris.feature_names)
df['especie'] = [iris.target_names[i] for i in y]
print("Primeras 5 filas del dataset:")
print(df.head())

# Entrenar un árbol de decisión individual
arbol_individual = DecisionTreeClassifier(max_depth=3, random_state=42)
arbol_individual.fit(X, y)

# Entrenar un Random Forest (muchos árboles)
bosque = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
bosque.fit(X, y)

# Comparar predicciones
ejemplo = X[0:1]  # Primera flor
pred_arbol = arbol_individual.predict(ejemplo)
pred_bosque = bosque.predict(ejemplo)
prob_bosque = bosque.predict_proba(ejemplo)

print(f"\nEjemplo de predicción:")
print(f"Características: {ejemplo[0]}")
print(f"Árbol individual predice: {iris.target_names[pred_arbol[0]]}")
print(f"Random Forest predice: {iris.target_names[pred_bosque[0]]}")
print(f"Probabilidades del Random Forest: {prob_bosque[0]}")

# Visualizar un árbol del bosque
plt.figure(figsize=(15, 10))
plot_tree(arbol_individual, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True)
plt.title("Ejemplo de Árbol de Decisión")
plt.show()

# Ver la importancia de las características
importancias = bosque.feature_importances_
caracteristicas = iris.feature_names

plt.figure(figsize=(10, 6))
plt.bar(caracteristicas, importancias)
plt.title('Importancia de las Características según Random Forest')
plt.ylabel('Importancia')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

#### 3. SVM (Support Vector Machine)

**¿Qué es?**
Imagina que tienes puntos rojos y azules en un papel, y quieres trazar la mejor línea para separarlos. SVM encuentra la línea que deja el mayor "margen" (espacio) entre ambos grupos.

```python
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Crear datos separables
X, y = make_blobs(n_samples=100, centers=2, n_features=2,
                 cluster_std=1.5, random_state=42)

# Entrenar SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X, y)

# Función para visualizar la frontera de decisión
def plot_svm_decision_boundary(model, X, y, title):
    plt.figure(figsize=(10, 8))

    # Crear malla de puntos
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Hacer predicciones en la malla
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotear
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='black')

    # Marcar los vectores de soporte
    support_vectors = model.support_vectors_
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
               s=200, facecolors='none', edgecolors='black', linewidth=2,
               label='Vectores de Soporte')

    plt.title(title)
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.legend()
    plt.show()

plot_svm_decision_boundary(svm_model, X, y, "SVM: Separación Lineal")

# Ejemplo con datos no linealmente separables
from sklearn.datasets import make_circles

# Crear datos circulares (no separables linealmente)
X_circle, y_circle = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=42)

# SVM con kernel lineal (no funcionará bien)
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_circle, y_circle)

# SVM con kernel RBF (funciona mejor para datos no lineales)
svm_rbf = SVC(kernel='rbf', gamma='scale')
svm_rbf.fit(X_circle, y_circle)

# Comparar resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Linear SVM
h = 0.02
x_min, x_max = X_circle[:, 0].min() - 1, X_circle[:, 0].max() + 1
y_min, y_max = X_circle[:, 1].min() - 1, X_circle[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z_linear = svm_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z_linear = Z_linear.reshape(xx.shape)

ax1.contourf(xx, yy, Z_linear, alpha=0.3, cmap=plt.cm.RdBu)
ax1.scatter(X_circle[:, 0], X_circle[:, 1], c=y_circle, cmap=plt.cm.RdBu, edgecolors='black')
ax1.set_title('SVM Lineal (No funciona bien)')
ax1.set_xlabel('Característica 1')
ax1.set_ylabel('Característica 2')

# RBF SVM
Z_rbf = svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rbf = Z_rbf.reshape(xx.shape)

ax2.contourf(xx, yy, Z_rbf, alpha=0.3, cmap=plt.cm.RdBu)
ax2.scatter(X_circle[:, 0], X_circle[:, 1], c=y_circle, cmap=plt.cm.RdBu, edgecolors='black')
ax2.set_title('SVM con Kernel RBF (Funciona mejor)')
ax2.set_xlabel('Característica 1')
ax2.set_ylabel('Característica 2')

plt.tight_layout()
plt.show()

print(f"Precisión SVM Lineal: {svm_linear.score(X_circle, y_circle):.2f}")
print(f"Precisión SVM RBF: {svm_rbf.score(X_circle, y_circle):.2f}")
```

### Problemas con Datos Desbalanceados

**¿Qué significa "desbalanceado"?**
Imagina que tienes un dataset para detectar fraude en tarjetas de crédito:
- 9,900 transacciones normales
- 100 transacciones fraudulentas

El modelo podría "hacer trampa" y predecir que TODAS las transacciones son normales, obteniendo 99% de precisión, ¡pero fallando completamente en detectar fraude!

#### Técnicas de Balanceo

```python
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import matplotlib.pyplot as plt

# Crear dataset desbalanceado
X_imb, y_imb = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1,
                                   weights=[0.95, 0.05], random_state=42)

print("Dataset original:")
print(f"Clase 0: {Counter(y_imb)[0]} muestras")
print(f"Clase 1: {Counter(y_imb)[1]} muestras")

# Técnica 1: Oversampling con SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_imb, y_imb)

print("\nDespués de SMOTE (Oversampling):")
print(f"Clase 0: {Counter(y_smote)[0]} muestras")
print(f"Clase 1: {Counter(y_smote)[1]} muestras")

# Técnica 2: Undersampling
undersampler = RandomUnderSampler(random_state=42)
X_under, y_under = undersampler.fit_resample(X_imb, y_imb)

print("\nDespués de Undersampling:")
print(f"Clase 0: {Counter(y_under)[0]} muestras")
print(f"Clase 1: {Counter(y_under)[1]} muestras")

# Visualizar las diferencias
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Dataset original
axes[0].scatter(X_imb[y_imb==0, 0], X_imb[y_imb==0, 1],
                c='red', marker='o', alpha=0.7, label='Clase 0')
axes[0].scatter(X_imb[y_imb==1, 0], X_imb[y_imb==1, 1],
                c='blue', marker='s', alpha=0.7, label='Clase 1')
axes[0].set_title('Dataset Original (Desbalanceado)')
axes[0].legend()

# Dataset con SMOTE
axes[1].scatter(X_smote[y_smote==0, 0], X_smote[y_smote==0, 1],
                c='red', marker='o', alpha=0.7, label='Clase 0')
axes[1].scatter(X_smote[y_smote==1, 0], X_smote[y_smote==1, 1],
                c='blue', marker='s', alpha=0.7, label='Clase 1')
axes[1].set_title('Después de SMOTE (Oversampling)')
axes[1].legend()

# Dataset con Undersampling
axes[2].scatter(X_under[y_under==0, 0], X_under[y_under==0, 1],
                c='red', marker='o', alpha=0.7, label='Clase 0')
axes[2].scatter(X_under[y_under==1, 0], X_under[y_under==1, 1],
                c='blue', marker='s', alpha=0.7, label='Clase 1')
axes[2].set_title('Después de Undersampling')
axes[2].legend()

plt.tight_layout()
plt.show()

# Comparar rendimiento de modelos
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Función para entrenar y evaluar
def entrenar_y_evaluar(X, y, nombre):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n--- {nombre} ---")
    print(classification_report(y_test, predictions))

entrenar_y_evaluar(X_imb, y_imb, "Dataset Original")
entrenar_y_evaluar(X_smote, y_smote, "Dataset con SMOTE")
entrenar_y_evaluar(X_under, y_under, "Dataset con Undersampling")
```

### Clasificación Multiclase

**¿Qué es?**
En lugar de solo 2 categorías (como spam/no spam), tenemos múltiples categorías (como perro/gato/pájaro/pez).

```python
from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset de dígitos (0-9)
digits = load_digits()
X, y = digits.data, digits.target

print(f"Tenemos {len(np.unique(y))} clases diferentes: {np.unique(y)}")
print(f"Forma de los datos: {X.shape}")

# Visualizar algunos ejemplos
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(10):
    # Encontrar el primer ejemplo de cada dígito
    idx = np.where(y == i)[0][0]
    axes[i].imshow(digits.images[idx], cmap='gray')
    axes[i].set_title(f'Dígito: {i}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Estrategia 1: One vs Rest (uno contra todos)
ovr_classifier = OneVsRestClassifier(SVC(kernel='rbf', gamma='scale'))
ovr_classifier.fit(X_train, y_train)
ovr_pred = ovr_classifier.predict(X_test)

# Estrategia 2: One vs One (uno contra uno)
ovo_classifier = OneVsOneClassifier(SVC(kernel='rbf', gamma='scale'))
ovo_classifier.fit(X_train, y_train)
ovo_pred = ovo_classifier.predict(X_test)

# Comparar resultados
print(f"Precisión One vs Rest: {accuracy_score(y_test, ovr_pred):.3f}")
print(f"Precisión One vs One: {accuracy_score(y_test, ovo_pred):.3f}")

# Matriz de confusión
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm_ovr = confusion_matrix(y_test, ovr_pred)
sns.heatmap(cm_ovr, annot=True, fmt='d', cmap='Blues')
plt.title('One vs Rest - Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')

plt.subplot(1, 2, 2)
cm_ovo = confusion_matrix(y_test, ovo_pred)
sns.heatmap(cm_ovo, annot=True, fmt='d', cmap='Blues')
plt.title('One vs One - Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')

plt.tight_layout()
plt.show()
```

### Funciones de Activación

**¿Qué son?**
Las funciones de activación son como "filtros" que deciden cómo transformar la información antes de pasarla al siguiente nivel del modelo.

```python
import numpy as np
import matplotlib.pyplot as plt

# Definir las funciones de activación
def sigmoid(x):
    """Convierte cualquier número a un valor entre 0 y 1"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """Si es negativo, devuelve 0. Si es positivo, devuelve el mismo valor"""
    return np.maximum(0, x)

def softmax(x):
    """Convierte un vector de números en probabilidades que suman 1"""
    exp_x = np.exp(x - np.max(x))  # Estabilidad numérica
    return exp_x / np.sum(exp_x)

# Crear datos para graficar
x = np.linspace(-6, 6, 1000)

# Calcular las funciones
y_sigmoid = sigmoid(x)
y_relu = relu(x)

# Para softmax, necesitamos un ejemplo específico
x_softmax = np.array([2.0, 1.0, 0.1])
y_softmax = softmax(x_softmax)

# Graficar
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Sigmoid
axes[0, 0].plot(x, y_sigmoid, 'b-', linewidth=3)
axes[0, 0].set_title('Función Sigmoid')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('sigmoid(x)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].text(-3, 0.8, 'Ideal para clasificación binaria\n(salida entre 0 y 1)',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

# ReLU
axes[0, 1].plot(x, y_relu, 'r-', linewidth=3)
axes[0, 1].set_title('Función ReLU')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('ReLU(x)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].text(-4, 4, 'Muy usada en redes neuronales\n(rápida de calcular)',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

# Softmax (ejemplo con barras)
clases = ['Perro', 'Gato', 'Pájaro']
axes[1, 0].bar(clases, y_softmax, color=['brown', 'orange', 'skyblue'])
axes[1, 0].set_title('Función Softmax')
axes[1, 0].set_ylabel('Probabilidad')
for i, v in enumerate(y_softmax):
    axes[1, 0].text(i, v + 0.01, f'{v:.2f}', ha='center')
axes[1, 0].text(1, 0.7, f'Suma total: {np.sum(y_softmax):.1f}\nIdeal para clasificación multiclase',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

# Ejemplo práctico: Clasificación de emociones
axes[1, 1].axis('off')
ejemplo_texto = """
EJEMPLO PRÁCTICO:

Clasificación de emociones en texto:
"¡Estoy muy feliz hoy!"

1. El modelo procesa el texto
2. Genera puntuaciones:
   - Feliz: 3.2
   - Triste: -1.1
   - Enojado: -0.8
   - Neutral: 0.5

3. Aplicamos Softmax:
   - Feliz: 0.71 (71%)
   - Triste: 0.09 (9%)
   - Enojado: 0.12 (12%)
   - Neutral: 0.08 (8%)

¡El modelo predice "Feliz" con 71% de confianza!
"""

axes[1, 1].text(0.05, 0.95, ejemplo_texto, transform=axes[1, 1].transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

plt.tight_layout()
plt.show()

# Ejemplo interactivo de funciones de activación
def comparar_activaciones(valor_entrada):
    """Compara cómo diferentes funciones procesan el mismo valor"""
    sigmoid_result = sigmoid(valor_entrada)
    relu_result = relu(valor_entrada)

    print(f"\nValor de entrada: {valor_entrada}")
    print(f"Sigmoid: {sigmoid_result:.4f}")
    print(f"ReLU: {relu_result:.4f}")
    print("-" * 30)

# Probar con diferentes valores
valores_test = [-5, -1, 0, 1, 5]
print("Comparación de funciones de activación:")
for valor in valores_test:
    comparar_activaciones(valor)

---

## Evaluación de Modelos

### ¿Por qué es Importante Evaluar Modelos?

Imagina que creas un modelo para diagnosticar enfermedades. No basta con que funcione en los datos que usaste para entrenarlo, ¡necesitas saber si funcionará con pacientes reales que nunca ha visto!

**Analogía del estudiante:**
- **Datos de entrenamiento** = Ejercicios que practica el estudiante
- **Datos de prueba** = El examen final (nunca visto antes)
- **Evaluación** = La calificación del examen

### Métricas de Evaluación

#### 1. Accuracy (Precisión General)

**¿Qué es?**
Es la proporción de predicciones correctas sobre el total.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Crear datos de ejemplo
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Calcular accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

# Ejemplo visual con casos reales
ejemplos_reales = [
    "Predicción: Spam, Real: Spam ✓",
    "Predicción: No Spam, Real: No Spam ✓",
    "Predicción: Spam, Real: No Spam ✗",
    "Predicción: No Spam, Real: Spam ✗",
    "Predicción: Spam, Real: Spam ✓"
]

print("\nEjemplo conceptual:")
correctos = 3
total = 5
print(f"De {total} predicciones, {correctos} fueron correctas")
print(f"Accuracy = {correctos}/{total} = {correctos/total:.2f} = {correctos/total*100:.0f}%")
```

#### 2. Matriz de Confusión

**¿Qué es?**
Es una tabla que muestra qué predijo el modelo vs qué era realmente correcto.

```python
# Matriz de confusión
cm = confusion_matrix(y_test, predictions)

# Visualizar matriz de confusión
plt.figure(figsize=(12, 5))

# Matriz de confusión simple
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicho: Clase 0', 'Predicho: Clase 1'],
            yticklabels=['Real: Clase 0', 'Real: Clase 1'])
plt.title('Matriz de Confusión')

# Ejemplo más detallado
plt.subplot(1, 2, 2)
# Crear matriz de ejemplo más clara
ejemplo_matriz = np.array([[85, 5], [10, 200]])
labels = ['No Spam', 'Spam']

sns.heatmap(ejemplo_matriz, annot=True, fmt='d', cmap='RdYlBu_r',
            xticklabels=[f'Predicho: {l}' for l in labels],
            yticklabels=[f'Real: {l}' for l in labels])
plt.title('Ejemplo: Detector de Spam')

# Añadir explicaciones
plt.figtext(0.02, 0.4, """
INTERPRETACIÓN:
• Verdaderos Negativos (85): Emails normales identificados correctamente
• Falsos Positivos (5): Emails normales marcados como spam ⚠️
• Falsos Negativos (10): Spam no detectado ⚠️⚠️ (¡MÁS GRAVE!)
• Verdaderos Positivos (200): Spam detectado correctamente
""", fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))

plt.tight_layout()
plt.show()
```

#### 3. Precision, Recall y F1-Score

**Explicación con analogía:**

Imagina un detector de incendios:
- **Precision**: De todas las alarmas que sonaron, ¿cuántas eran incendios reales?
- **Recall**: De todos los incendios reales, ¿cuántos detectó?
- **F1-Score**: Balance entre precision y recall

```python
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

# Calcular métricas
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("MÉTRICAS DETALLADAS:")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# Explicación visual con ejemplos
def explicar_metricas_con_ejemplos():
    # Ejemplo: Sistema de detección de fraude
    print("\n" + "="*50)
    print("EJEMPLO: DETECTOR DE FRAUDE EN TARJETAS")
    print("="*50)

    # Datos simulados
    total_transacciones = 1000
    fraudes_reales = 50
    fraudes_detectados = 40
    falsas_alarmas = 10

    # Matriz de confusión conceptual
    vn = total_transacciones - fraudes_reales - falsas_alarmas  # Verdaderos negativos
    vp = fraudes_detectados - (fraudes_detectados - (fraudes_reales - (fraudes_reales - fraudes_detectados)))  # Verdaderos positivos
    fp = falsas_alarmas  # Falsos positivos
    fn = fraudes_reales - fraudes_detectados  # Falsos negativos

    # Ajustar números para que sean consistentes
    vp = 35  # Fraudes correctamente detectados
    fn = 15  # Fraudes no detectados
    fp = 10  # Transacciones normales marcadas como fraude
    vn = 940 # Transacciones normales correctamente identificadas

    print(f"📊 RESULTADOS:")
    print(f"• Verdaderos Positivos (VP): {vp} - Fraudes detectados correctamente")
    print(f"• Falsos Negativos (FN): {fn} - Fraudes no detectados ⚠️")
    print(f"• Falsos Positivos (FP): {fp} - Falsas alarmas")
    print(f"• Verdaderos Negativos (VN): {vn} - Transacciones normales correctas")

    # Calcular métricas
    precision_ej = vp / (vp + fp)
    recall_ej = vp / (vp + fn)
    f1_ej = 2 * (precision_ej * recall_ej) / (precision_ej + recall_ej)

    print(f"\n📈 MÉTRICAS:")
    print(f"• Precision = VP/(VP+FP) = {vp}/({vp}+{fp}) = {precision_ej:.2f}")
    print(f"  → De cada 100 alarmas, {precision_ej*100:.0f} son fraudes reales")

    print(f"• Recall = VP/(VP+FN) = {vp}/({vp}+{fn}) = {recall_ej:.2f}")
    print(f"  → Detectamos {recall_ej*100:.0f}% de todos los fraudes")

    print(f"• F1-Score = {f1_ej:.2f}")
    print(f"  → Balance entre precision y recall")

    print(f"\n🤔 INTERPRETACIÓN:")
    if precision_ej > 0.8:
        print("✅ Precision alta: Pocas falsas alarmas")
    else:
        print("❌ Precision baja: Muchas falsas alarmas")

    if recall_ej > 0.8:
        print("✅ Recall alto: Detectamos la mayoría de fraudes")
    else:
        print("❌ Recall bajo: Se nos escapan muchos fraudes")

explicar_metricas_con_ejemplos()

# Gráfico de Precision-Recall
y_scores = model.predict_proba(X_test)[:, 1]  # Probabilidades de la clase positiva
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(recall_vals, precision_vals, 'b-', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.grid(True, alpha=0.3)

# Marcar punto actual
current_precision = precision_score(y_test, predictions)
current_recall = recall_score(y_test, predictions)
plt.plot(current_recall, current_precision, 'ro', markersize=10, label='Modelo Actual')
plt.legend()

# Comparación de umbrales
plt.subplot(1, 2, 2)
plt.plot(thresholds, precision_vals[:-1], 'g-', label='Precision', linewidth=2)
plt.plot(thresholds, recall_vals[:-1], 'r-', label='Recall', linewidth=2)
plt.xlabel('Umbral de Decisión')
plt.ylabel('Valor')
plt.title('Precision vs Recall por Umbral')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 4. Curva ROC

**¿Qué es?**
La curva ROC muestra qué tan bien el modelo distingue entre clases en todos los umbrales posibles.

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Calcular curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
auc_score = roc_auc_score(y_test, y_scores)

plt.figure(figsize=(15, 5))

# Curva ROC
plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'r--', label='Modelo Aleatorio')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend()
plt.grid(True, alpha=0.3)

# Explicación visual del AUC
plt.subplot(1, 3, 2)
plt.fill_between(fpr, tpr, alpha=0.3, color='blue')
plt.plot(fpr, tpr, 'b-', linewidth=2)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title(f'Área Bajo la Curva = {auc_score:.2f}')
plt.grid(True, alpha=0.3)

# Interpretación del AUC
plt.subplot(1, 3, 3)
plt.axis('off')

interpretacion_auc = f"""
INTERPRETACIÓN DEL AUC:

AUC = {auc_score:.2f}

🎯 ESCALA:
• 1.0 = Modelo perfecto
• 0.9-1.0 = Excelente
• 0.8-0.9 = Bueno
• 0.7-0.8 = Aceptable
• 0.6-0.7 = Pobre
• 0.5 = Modelo aleatorio
• < 0.5 = Peor que aleatorio

📊 SIGNIFICADO:
El AUC representa la probabilidad
de que el modelo asigne una
puntuación más alta a un ejemplo
positivo aleatorio que a un
ejemplo negativo aleatorio.

Con AUC = {auc_score:.2f}, nuestro modelo
tiene un rendimiento {"excelente" if auc_score > 0.9 else "bueno" if auc_score > 0.8 else "aceptable" if auc_score > 0.7 else "pobre"}.
"""

plt.text(0.05, 0.95, interpretacion_auc, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))

plt.tight_layout()
plt.show()
```

### Overfitting y Underfitting

**¿Qué son?**
- **Underfitting**: El modelo es demasiado simple (como estudiar solo 1 hora para un examen de 100 temas)
- **Overfitting**: El modelo memoriza en lugar de aprender (como memorizar solo las preguntas del examen de práctica)

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

# Generar datos con ruido
np.random.seed(42)
n_samples = 100
X_simple = np.linspace(0, 1, n_samples).reshape(-1, 1)
y_true = 1.5 * X_simple.ravel() + 0.5 * np.sin(15 * X_simple.ravel())
y_simple = y_true + np.random.normal(0, 0.1, n_samples)

# Crear modelos con diferente complejidad
grados = [1, 4, 15]
colores = ['red', 'green', 'blue']
nombres = ['Underfitting\n(Muy Simple)', 'Buen Ajuste\n(Equilibrado)', 'Overfitting\n(Muy Complejo)']

plt.figure(figsize=(15, 10))

# Gráfico principal: comparación de modelos
plt.subplot(2, 2, 1)
plt.scatter(X_simple, y_simple, alpha=0.6, color='gray', s=20, label='Datos con ruido')
plt.plot(X_simple, y_true, 'black', linewidth=2, label='Función verdadera')

X_plot = np.linspace(0, 1, 300).reshape(-1, 1)

for i, (grado, color, nombre) in enumerate(zip(grados, colores, nombres)):
    # Entrenar modelo polinomial
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=grado)),
        ('linear', LinearRegression())
    ])
    poly_model.fit(X_simple, y_simple)
    y_plot = poly_model.predict(X_plot)

    plt.plot(X_plot, y_plot, color=color, linewidth=2, label=f'{nombre} (grado {grado})')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparación: Underfitting vs Overfitting')
plt.legend()
plt.grid(True, alpha=0.3)

# Análisis de error vs complejidad
grados_test = range(1, 16)
train_errors = []
val_errors = []

# Dividir datos
X_train, X_val, y_train, y_val = train_test_split(X_simple, y_simple, test_size=0.3, random_state=42)

for grado in grados_test:
    # Entrenar modelo
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=grado)),
        ('linear', LinearRegression())
    ])
    poly_model.fit(X_train, y_train)

    # Calcular errores
    train_pred = poly_model.predict(X_train)
    val_pred = poly_model.predict(X_val)

    train_error = np.mean((train_pred - y_train) ** 2)
    val_error = np.mean((val_pred - y_val) ** 2)

    train_errors.append(train_error)
    val_errors.append(val_error)

# Gráfico de curvas de aprendizaje
plt.subplot(2, 2, 2)
plt.plot(grados_test, train_errors, 'o-', color='blue', label='Error Entrenamiento')
plt.plot(grados_test, val_errors, 'o-', color='red', label='Error Validación')
plt.xlabel('Grado del Polinomio (Complejidad)')
plt.ylabel('Error Cuadrático Medio')
plt.title('Curvas de Validación')
plt.legend()
plt.grid(True, alpha=0.3)

# Marcar zonas
plt.axvspan(1, 3, alpha=0.2, color='red', label='Underfitting')
plt.axvspan(3, 6, alpha=0.2, color='green', label='Buen ajuste')
plt.axvspan(6, 15, alpha=0.2, color='blue', label='Overfitting')

# Explicación detallada
plt.subplot(2, 2, 3)
plt.axis('off')
explicacion = """
🔍 DIAGNÓSTICO DE MODELOS:

📈 UNDERFITTING (Grado 1):
• Error alto en entrenamiento y validación
• El modelo es demasiado simple
• Solución: Aumentar complejidad

⚖️ BUEN AJUSTE (Grado 4):
• Error bajo y similar en entrenamiento y validación
• El modelo generaliza bien
• ¡Ideal!

📉 OVERFITTING (Grado 15):
• Error muy bajo en entrenamiento
• Error alto en validación
• El modelo memoriza en lugar de aprender
• Solución: Reducir complejidad o más datos
"""

plt.text(0.05, 0.95, explicacion, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

# Ejemplo con datos de tamaño variable
plt.subplot(2, 2, 4)
tamaños = np.array([10, 20, 50, 100, 200, 500])
train_scores_mean = []
val_scores_mean = []

for tamaño in tamaños:
    if tamaño <= len(X_simple):
        # Usar subconjunto de datos
        X_sub = X_simple[:tamaño]
        y_sub = y_simple[:tamaño]

        # Modelo de complejidad media
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=4)),
            ('linear', LinearRegression())
        ])

        # Validación cruzada simple
        scores_train = []
        scores_val = []

        for _ in range(5):  # 5 divisiones aleatorias
            X_t, X_v, y_t, y_v = train_test_split(X_sub, y_sub, test_size=0.3)
            model.fit(X_t, y_t)

            scores_train.append(model.score(X_t, y_t))
            scores_val.append(model.score(X_v, y_v))

        train_scores_mean.append(np.mean(scores_train))
        val_scores_mean.append(np.mean(scores_val))

if train_scores_mean:  # Solo si tenemos datos
    plt.plot(tamaños[:len(train_scores_mean)], train_scores_mean, 'o-',
             color='blue', label='Score Entrenamiento')
    plt.plot(tamaños[:len(val_scores_mean)], val_scores_mean, 'o-',
             color='red', label='Score Validación')

    plt.xlabel('Tamaño del Dataset')
    plt.ylabel('R² Score')
    plt.title('Curva de Aprendizaje')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Regularización

**¿Qué es?**
Es como poner "límites" al modelo para evitar que sea demasiado complejo.

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# Generar datos con muchas características
X_reg, y_reg = make_regression(n_samples=100, n_features=20, noise=10, random_state=42)

# Estandarizar datos
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)

# Dividir datos
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_scaled, y_reg, test_size=0.3, random_state=42)

# Probar diferentes valores de regularización
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Modelos de regularización
modelos = {
    'Ridge (L2)': Ridge,
    'Lasso (L1)': Lasso,
    'ElasticNet (L1+L2)': ElasticNet
}

plt.figure(figsize=(15, 10))

for i, (nombre, modelo_class) in enumerate(modelos.items()):
    plt.subplot(2, 3, i + 1)

    train_scores = []
    test_scores = []

    for alpha in alphas:
        if modelo_class == ElasticNet:
            model = modelo_class(alpha=alpha, l1_ratio=0.5, random_state=42)
        else:
            model = modelo_class(alpha=alpha, random_state=42)

        model.fit(X_train_reg, y_train_reg)

        train_score = model.score(X_train_reg, y_train_reg)
        test_score = model.score(X_test_reg, y_test_reg)

        train_scores.append(train_score)
        test_scores.append(test_score)

    plt.semilogx(alphas, train_scores, 'o-', label='Entrenamiento', color='blue')
    plt.semilogx(alphas, test_scores, 'o-', label='Validación', color='red')
    plt.xlabel('Parámetro de Regularización (α)')
    plt.ylabel('R² Score')
    plt.title(f'{nombre}')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Comparar coeficientes
plt.subplot(2, 3, 4)
# Modelo sin regularización
model_normal = LinearRegression()
model_normal.fit(X_train_reg, y_train_reg)

# Modelos regularizados
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train_reg, y_train_reg)

model_lasso = Lasso(alpha=1.0)
model_lasso.fit(X_train_reg, y_train_reg)

# Graficar coeficientes
caracteristicas = range(len(model_normal.coef_))
plt.plot(caracteristicas, model_normal.coef_, 'o-', label='Sin Regularización', linewidth=2)
plt.plot(caracteristicas, model_ridge.coef_, 'o-', label='Ridge (L2)', linewidth=2)
plt.plot(caracteristicas, model_lasso.coef_, 'o-', label='Lasso (L1)', linewidth=2)
plt.xlabel('Índice de Característica')
plt.ylabel('Valor del Coeficiente')
plt.title('Comparación de Coeficientes')
plt.legend()
plt.grid(True, alpha=0.3)

# Explicación de regularización
plt.subplot(2, 3, 5)
plt.axis('off')
explicacion_reg = """
🎯 TIPOS DE REGULARIZACIÓN:

📊 RIDGE (L2):
• Penaliza la suma de cuadrados de coeficientes
• Reduce todos los coeficientes proporcionalmente
• Nunca hace coeficientes exactamente cero
• Bueno cuando todas las características importan

📈 LASSO (L1):
• Penaliza la suma de valores absolutos
• Puede hacer coeficientes exactamente cero
• Selección automática de características
• Bueno para datasets con muchas características irrelevantes

⚖️ ELASTIC NET:
• Combinación de L1 y L2
• Balance entre Ridge y Lasso
• l1_ratio controla la mezcla
"""

plt.text(0.05, 0.95, explicacion_reg, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))

# Ejemplo práctico de selección de características con Lasso
plt.subplot(2, 3, 6)
lasso_path = Lasso(alpha=1.0)
lasso_path.fit(X_train_reg, y_train_reg)

caracteristicas_seleccionadas = np.where(np.abs(lasso_path.coef_) > 0.01)[0]
caracteristicas_eliminadas = np.where(np.abs(lasso_path.coef_) <= 0.01)[0]

plt.bar(range(len(lasso_path.coef_)), np.abs(lasso_path.coef_),
        color=['green' if i in caracteristicas_seleccionadas else 'red'
               for i in range(len(lasso_path.coef_))])
plt.xlabel('Índice de Característica')
plt.ylabel('|Coeficiente|')
plt.title('Selección de Características con Lasso')
plt.axhline(y=0.01, color='black', linestyle='--', label='Umbral')

# Añadir texto explicativo
plt.text(0.7, 0.8, f'Seleccionadas: {len(caracteristicas_seleccionadas)}\nEliminadas: {len(caracteristicas_eliminadas)}',
         transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n🎯 RESULTADOS DE REGULARIZACIÓN:")
print(f"Ridge R²: {model_ridge.score(X_test_reg, y_test_reg):.3f}")
print(f"Lasso R²: {model_lasso.score(X_test_reg, y_test_reg):.3f}")
print(f"Características seleccionadas por Lasso: {len(caracteristicas_seleccionadas)}/{len(lasso_path.coef_)}")
```

---

## Validación Cruzada

### ¿Qué es la Validación Cruzada?

**Analogía del examen:**
Imagina que quieres saber qué tan bien estudia un estudiante. En lugar de hacer solo 1 examen (que podría ser muy fácil o muy difícil por casualidad), haces 5 exámenes diferentes y calculas el promedio. ¡Así tienes una evaluación más confiable!

La **validación cruzada** hace exactamente esto con los datos: divide el dataset en varias partes y evalúa el modelo múltiples veces para obtener una estimación más robusta del rendimiento.

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, cross_validate
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Crear dataset de ejemplo
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                         n_informative=5, n_redundant=2, random_state=42)

print("🎯 EJEMPLO CONCEPTUAL DE VALIDACIÓN CRUZADA:")
print("=" * 60)

# Ejemplo visual de cómo funciona k-fold
def explicar_k_fold():
    print("\n📊 K-FOLD CROSS VALIDATION (k=5):")
    print("\nDatos: [1][2][3][4][5][6][7][8][9][10] (10 ejemplos)")
    print("\nIteración 1: Entrenamiento=[2][3][4][5][6][7][8][9][10] | Test=[1]")
    print("Iteración 2: Entrenamiento=[1][3][4][5][6][7][8][9][10] | Test=[2]")
    print("Iteración 3: Entrenamiento=[1][2][4][5][6][7][8][9][10] | Test=[3]")
    print("Iteración 4: Entrenamiento=[1][2][3][5][6][7][8][9][10] | Test=[4]")
    print("Iteración 5: Entrenamiento=[1][2][3][4][6][7][8][9][10] | Test=[5]")
    print("\n📈 Resultado final = Promedio de los 5 scores")

explicar_k_fold()

# Validación cruzada básica
model = RandomForestClassifier(random_state=42)

# K-Fold Cross Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"\n🎯 RESULTADOS DE VALIDACIÓN CRUZADA:")
print(f"Scores individuales: {[f'{score:.3f}' for score in cv_scores]}")
print(f"Score promedio: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"Intervalo de confianza (95%): [{cv_scores.mean() - 2*cv_scores.std():.3f}, {cv_scores.mean() + 2*cv_scores.std():.3f}]")
```

### Tipos de Validación Cruzada

```python
# Comparar diferentes tipos de validación cruzada
from sklearn.model_selection import ShuffleSplit, LeaveOneOut

# Crear datos con clases desbalanceadas para mostrar la diferencia
X_imb, y_imb = make_classification(n_samples=300, n_classes=2, weights=[0.9, 0.1],
                                  n_features=10, random_state=42)

print(f"\n📊 DATASET DESBALANCEADO:")
print(f"Clase 0: {np.sum(y_imb == 0)} muestras ({np.sum(y_imb == 0)/len(y_imb)*100:.1f}%)")
print(f"Clase 1: {np.sum(y_imb == 1)} muestras ({np.sum(y_imb == 1)/len(y_imb)*100:.1f}%)")

# Diferentes estrategias de validación cruzada
estrategias_cv = {
    'KFold': KFold(n_splits=5, shuffle=True, random_state=42),
    'StratifiedKFold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'ShuffleSplit': ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
}

model = LogisticRegression(random_state=42)
resultados = {}

print(f"\n🔍 COMPARACIÓN DE ESTRATEGIAS DE CV:")
print("-" * 60)

for nombre, cv_strategy in estrategias_cv.items():
    scores = cross_val_score(model, X_imb, y_imb, cv=cv_strategy, scoring='accuracy')
    resultados[nombre] = scores

    print(f"{nombre:15} | Promedio: {scores.mean():.3f} ± {scores.std():.3f}")

# Visualizar diferencias
plt.figure(figsize=(15, 10))

# Gráfico 1: Comparación de estrategias
plt.subplot(2, 3, 1)
nombres = list(resultados.keys())
promedios = [scores.mean() for scores in resultados.values()]
stds = [scores.std() for scores in resultados.values()]

bars = plt.bar(nombres, promedios, yerr=stds, capsize=5, alpha=0.7, color=['skyblue', 'lightgreen', 'coral'])
plt.title('Comparación de Estrategias de CV')
plt.ylabel('Accuracy Score')
plt.xticks(rotation=45)

# Añadir valores en las barras
for bar, promedio, std in zip(bars, promedios, stds):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
             f'{promedio:.3f}', ha='center', va='bottom')

plt.grid(True, alpha=0.3)

# Gráfico 2: Distribución de scores
plt.subplot(2, 3, 2)
data_for_boxplot = [resultados[nombre] for nombre in nombres]
plt.boxplot(data_for_boxplot, labels=nombres)
plt.title('Distribución de Scores')
plt.ylabel('Accuracy Score')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Gráfico 3: Ejemplo visual de StratifiedKFold
plt.subplot(2, 3, 3)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_distributions = []
for fold, (train_idx, test_idx) in enumerate(skf.split(X_imb, y_imb)):
    test_class_0 = np.sum(y_imb[test_idx] == 0)
    test_class_1 = np.sum(y_imb[test_idx] == 1)
    total_test = len(test_idx)

    fold_distributions.append([test_class_0/total_test, test_class_1/total_test])

fold_distributions = np.array(fold_distributions)

x = np.arange(5)  # 5 folds
width = 0.35

plt.bar(x - width/2, fold_distributions[:, 0], width, label='Clase 0', alpha=0.7)
plt.bar(x + width/2, fold_distributions[:, 1], width, label='Clase 1', alpha=0.7)

plt.title('StratifiedKFold: Distribución por Fold')
plt.xlabel('Fold')
plt.ylabel('Proporción')
plt.xticks(x, [f'Fold {i+1}' for i in x])
plt.legend()
plt.grid(True, alpha=0.3)

# Explicación detallada
plt.subplot(2, 3, 4)
plt.axis('off')
explicacion_cv = """
📚 TIPOS DE VALIDACIÓN CRUZADA:

🔀 K-FOLD:
• Divide datos en k partes iguales
• Cada parte se usa como test una vez
• Simple pero puede crear desbalance

⚖️ STRATIFIED K-FOLD:
• Mantiene la proporción de clases
• Ideal para datos desbalanceados
• Más representativo

🎲 SHUFFLE SPLIT:
• Divisiones aleatorias múltiples
• Control del tamaño de test
• Más variabilidad en las muestras
"""

plt.text(0.05, 0.95, explicacion_cv, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan"))

plt.tight_layout()
plt.show()
```

### Validación Cruzada con Múltiples Métricas

```python
# Validación cruzada con múltiples métricas
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

print("\n📊 VALIDACIÓN CRUZADA CON MÚLTIPLES MÉTRICAS:")
print("=" * 60)

# Usar cross_validate para múltiples métricas
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring_metrics,
                          return_train_score=True)

# Mostrar resultados detallados
metricas_test = {}
metricas_train = {}

for metric in scoring_metrics:
    test_scores = cv_results[f'test_{metric}']
    train_scores = cv_results[f'train_{metric}']

    metricas_test[metric] = test_scores
    metricas_train[metric] = train_scores

    print(f"\n{metric.upper():15}")
    print(f"  Test:  {test_scores.mean():.3f} ± {test_scores.std():.3f}")
    print(f"  Train: {train_scores.mean():.3f} ± {train_scores.std():.3f}")

    # Detectar posible overfitting
    gap = train_scores.mean() - test_scores.mean()
    if gap > 0.05:  # Gap significativo
        print(f"  ⚠️  Posible overfitting (gap: {gap:.3f})")
    else:
        print(f"  ✅ Buen balance (gap: {gap:.3f})")

# Visualizar métricas
plt.figure(figsize=(15, 8))

# Gráfico de barras comparativo
plt.subplot(2, 2, 1)
x_pos = np.arange(len(scoring_metrics))
test_means = [metricas_test[m].mean() for m in scoring_metrics]
test_stds = [metricas_test[m].std() for m in scoring_metrics]
train_means = [metricas_train[m].mean() for m in scoring_metrics]
train_stds = [metricas_train[m].std() for m in scoring_metrics]

width = 0.35
plt.bar(x_pos - width/2, test_means, width, yerr=test_stds, label='Test',
        alpha=0.7, capsize=5)
plt.bar(x_pos + width/2, train_means, width, yerr=train_stds, label='Train',
        alpha=0.7, capsize=5)

plt.xlabel('Métricas')
plt.ylabel('Score')
plt.title('Comparación Train vs Test')
plt.xticks(x_pos, scoring_metrics, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico de radar
plt.subplot(2, 2, 2)
angles = np.linspace(0, 2 * np.pi, len(scoring_metrics), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))  # Cerrar el círculo

test_means_norm = np.concatenate((test_means, [test_means[0]]))
train_means_norm = np.concatenate((train_means, [train_means[0]]))

ax = plt.subplot(2, 2, 2, projection='polar')
ax.plot(angles, test_means_norm, 'o-', linewidth=2, label='Test', color='blue')
ax.plot(angles, train_means_norm, 'o-', linewidth=2, label='Train', color='red')
ax.fill(angles, test_means_norm, alpha=0.25, color='blue')
ax.fill(angles, train_means_norm, alpha=0.25, color='red')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(scoring_metrics)
ax.set_ylim(0, 1)
ax.set_title('Radar de Métricas', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Box plot de todas las métricas
plt.subplot(2, 2, 3)
all_test_data = [metricas_test[m] for m in scoring_metrics]
bp = plt.boxplot(all_test_data, labels=scoring_metrics, patch_artist=True)

# Colorear cada box
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.title('Distribución de Scores de Test')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Matriz de correlación entre métricas
plt.subplot(2, 2, 4)
# Crear DataFrame con todas las métricas
import pandas as pd
df_metricas = pd.DataFrame({metric: metricas_test[metric] for metric in scoring_metrics})
correlation_matrix = df_metricas.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
           square=True, cbar_kws={'shrink': 0.8})
plt.title('Correlación entre Métricas')
plt.tight_layout()
plt.show()
```

### Curvas de Aprendizaje y Validación

```python
# Curvas de aprendizaje
def plot_learning_curves():
    """Genera curvas de aprendizaje para diagnosticar el modelo"""

    # Cargar dataset más complejo
    breast_cancer = load_breast_cancer()
    X_bc, y_bc = breast_cancer.data, breast_cancer.target

    plt.figure(figsize=(15, 10))

    # Curva de aprendizaje
    plt.subplot(2, 2, 1)
    train_sizes, train_scores, val_scores = learning_curve(
        RandomForestClassifier(n_estimators=50, random_state=42),
        X_bc, y_bc, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    # Calcular medias y desviaciones
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plotear
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Entrenamiento')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='blue')

    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validación')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='red')

    plt.xlabel('Tamaño del Dataset de Entrenamiento')
    plt.ylabel('Accuracy Score')
    plt.title('Curva de Aprendizaje')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Curva de validación para n_estimators
    plt.subplot(2, 2, 2)
    param_range = [10, 25, 50, 100, 200, 300]
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(random_state=42), X_bc, y_bc,
        param_name='n_estimators', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.semilogx(param_range, train_mean, 'o-', color='blue', label='Entrenamiento')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='blue')

    plt.semilogx(param_range, val_mean, 'o-', color='red', label='Validación')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='red')

    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy Score')
    plt.title('Curva de Validación: n_estimators')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Curva de validación para max_depth
    plt.subplot(2, 2, 3)
    param_range = range(1, 21)
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(n_estimators=100, random_state=42), X_bc, y_bc,
        param_name='max_depth', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.plot(param_range, train_mean, 'o-', color='blue', label='Entrenamiento')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='blue')

    plt.plot(param_range, val_mean, 'o-', color='red', label='Validación')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='red')

    plt.xlabel('max_depth')
    plt.ylabel('Accuracy Score')
    plt.title('Curva de Validación: max_depth')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Interpretación
    plt.subplot(2, 2, 4)
    plt.axis('off')
    interpretacion = """
    🔍 INTERPRETACIÓN DE CURVAS:

    📈 CURVA DE APRENDIZAJE:
    • Si ambas curvas convergen: Buen modelo
    • Si gap grande: Overfitting
    • Si ambas bajas: Underfitting
    • Si mejora con más datos: Obtener más datos

    📊 CURVAS DE VALIDACIÓN:
    • Punto óptimo donde val es máximo
    • Después del óptimo: Overfitting
    • Antes del óptimo: Underfitting

    🎯 DIAGNÓSTICO ACTUAL:
    • n_estimators: Más árboles = mejor (hasta cierto punto)
    • max_depth: Cuidado con profundidad excesiva
    """

    plt.text(0.05, 0.95, interpretacion, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.tight_layout()
    plt.show()

plot_learning_curves()

print(f"\n✅ RESUMEN DE VALIDACIÓN CRUZADA:")
print(f"• La validación cruzada da una estimación más robusta del rendimiento")
print(f"• StratifiedKFold es mejor para datos desbalanceados")
print(f"• Las curvas de aprendizaje ayudan a diagnosticar problemas")
print(f"• Las curvas de validación ayudan a encontrar hiperparámetros óptimos")
```

---

## GridSearchCV

### ¿Qué es GridSearchCV?

**Analogía del chef:**
Imagina que eres un chef y quieres hacer la pizza perfecta. Tienes que decidir:
- ¿Cuántos minutos hornear? (10, 15, 20, 25)
- ¿Qué temperatura? (200°, 220°, 240°)
- ¿Cuánta cantidad de cada ingrediente?

En lugar de probar al azar, haces TODAS las combinaciones posibles y eliges la que mejor sabe. **GridSearchCV** hace exactamente esto, pero con parámetros de modelos de machine learning.

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from time import time

print("🔍 GRIDSEARCHCV: BÚSQUEDA AUTOMÁTICA DE HIPERPARÁMETROS")
print("=" * 70)

# Crear dataset de ejemplo
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                         n_redundant=3, n_classes=3, random_state=42)

print(f"📊 Dataset: {X.shape[0]} muestras, {X.shape[1]} características, {len(np.unique(y))} clases")

# Ejemplo básico con RandomForest
print(f"\n🌳 EJEMPLO 1: OPTIMIZACIÓN DE RANDOM FOREST")
print("-" * 50)

# Definir el espacio de búsqueda
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Calcular total de combinaciones
total_combinations = 1
for param, values in param_grid_rf.items():
    total_combinations *= len(values)
    print(f"  {param}: {values} ({len(values)} opciones)")

print(f"\n🔢 Total de combinaciones: {total_combinations}")
print(f"Con CV=5, total de entrenamientos: {total_combinations * 5} = {total_combinations * 5}")

# Ejecutar GridSearchCV
rf_base = RandomForestClassifier(random_state=42)

print(f"\n⏳ Ejecutando GridSearchCV...")
start_time = time()

grid_search_rf = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,  # Usar todos los cores
    verbose=1    # Mostrar progreso
)

grid_search_rf.fit(X, y)
end_time = time()

print(f"\n🎯 RESULTADOS:")
print(f"⏱️  Tiempo total: {end_time - start_time:.2f} segundos")
print(f"🏆 Mejor score: {grid_search_rf.best_score_:.4f}")
print(f"⚙️  Mejores parámetros:")
for param, value in grid_search_rf.best_params_.items():
    print(f"    {param}: {value}")

# Comparar con parámetros por defecto
rf_default = RandomForestClassifier(random_state=42)
from sklearn.model_selection import cross_val_score
default_score = cross_val_score(rf_default, X, y, cv=5, scoring='accuracy').mean()

print(f"\n📈 COMPARACIÓN:")
print(f"Parámetros por defecto: {default_score:.4f}")
print(f"Parámetros optimizados: {grid_search_rf.best_score_:.4f}")
print(f"Mejora: {grid_search_rf.best_score_ - default_score:.4f} ({((grid_search_rf.best_score_ - default_score) / default_score * 100):.1f}%)")
```

### Análisis de Resultados de GridSearchCV

```python
# Analizar todos los resultados
results_df = pd.DataFrame(grid_search_rf.cv_results_)

print(f"\n📊 ANÁLISIS DETALLADO DE RESULTADOS:")
print(f"Total de combinaciones probadas: {len(results_df)}")

# Top 10 mejores combinaciones
print(f"\n🏆 TOP 10 MEJORES COMBINACIONES:")
top_10 = results_df.nlargest(10, 'mean_test_score')[['mean_test_score', 'std_test_score', 'params']]
for i, (_, row) in enumerate(top_10.iterrows(), 1):
    print(f"{i:2d}. Score: {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
    print(f"    Parámetros: {row['params']}")

# Visualizaciones
plt.figure(figsize=(20, 12))

# Heatmap de n_estimators vs max_depth
plt.subplot(3, 3, 1)
pivot_data = results_df.groupby(['param_n_estimators', 'param_max_depth'])['mean_test_score'].mean().unstack()
sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis')
plt.title('Score vs n_estimators y max_depth')
plt.xlabel('max_depth')
plt.ylabel('n_estimators')

# Distribución de scores
plt.subplot(3, 3, 2)
plt.hist(results_df['mean_test_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(grid_search_rf.best_score_, color='red', linestyle='--',
           label=f'Mejor: {grid_search_rf.best_score_:.4f}')
plt.xlabel('Mean Test Score')
plt.ylabel('Frecuencia')
plt.title('Distribución de Scores')
plt.legend()
plt.grid(True, alpha=0.3)

# Efecto de cada parámetro
plt.subplot(3, 3, 3)
param_effects = {}
for param in ['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf']:
    if param in results_df.columns:
        grouped = results_df.groupby(param)['mean_test_score'].mean()
        param_effects[param.replace('param_', '')] = grouped

colors = ['blue', 'red', 'green', 'orange']
for i, (param_name, values) in enumerate(param_effects.items()):
    plt.plot(range(len(values)), values.values, 'o-',
            label=param_name, color=colors[i % len(colors)])

plt.xlabel('Posición del Parámetro')
plt.ylabel('Score Promedio')
plt.title('Efecto de cada Parámetro')
plt.legend()
plt.grid(True, alpha=0.3)

# Box plot por n_estimators
plt.subplot(3, 3, 4)
n_est_data = [results_df[results_df['param_n_estimators'] == n]['mean_test_score']
              for n in sorted(results_df['param_n_estimators'].unique())]
plt.boxplot(n_est_data, labels=sorted(results_df['param_n_estimators'].unique()))
plt.xlabel('n_estimators')
plt.ylabel('Score')
plt.title('Distribución por n_estimators')
plt.grid(True, alpha=0.3)

# Tiempo de ejecución vs score
plt.subplot(3, 3, 5)
plt.scatter(results_df['mean_fit_time'], results_df['mean_test_score'],
           alpha=0.6, c=results_df['mean_test_score'], cmap='viridis')
plt.xlabel('Tiempo de Entrenamiento (s)')
plt.ylabel('Score')
plt.title('Tiempo vs Rendimiento')
plt.colorbar(label='Score')
plt.grid(True, alpha=0.3)

# Análisis de estabilidad (std vs mean)
plt.subplot(3, 3, 6)
plt.scatter(results_df['mean_test_score'], results_df['std_test_score'],
           alpha=0.6, c='coral')
plt.xlabel('Score Promedio')
plt.ylabel('Desviación Estándar')
plt.title('Estabilidad de Resultados')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### GridSearchCV vs RandomizedSearchCV

```python
# Comparar GridSearchCV con RandomizedSearchCV
print(f"\n⚡ GRIDSEARCHCV VS RANDOMIZEDSEARCHCV")
print("=" * 60)

# Definir un espacio de búsqueda más grande para SVM
param_grid_svm_large = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Calcular combinaciones
total_svm = len(param_grid_svm_large['C']) * len(param_grid_svm_large['gamma']) * len(param_grid_svm_large['kernel'])
print(f"🔢 Total de combinaciones SVM: {total_svm}")

# Dataset más pequeño para que sea manejable
X_small, y_small = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

# Estandarizar datos para SVM
scaler = StandardScaler()
X_small_scaled = scaler.fit_transform(X_small)

# GridSearchCV con SVM (limitado)
param_grid_svm_limited = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

print(f"\n🔍 GridSearchCV (limitado):")
start_time = time()
grid_svm = GridSearchCV(SVC(random_state=42), param_grid_svm_limited, cv=3, n_jobs=-1)
grid_svm.fit(X_small_scaled, y_small)
grid_time = time() - start_time

print(f"⏱️  Tiempo: {grid_time:.2f}s")
print(f"🎯 Mejor score: {grid_svm.best_score_:.4f}")
print(f"⚙️  Mejores parámetros: {grid_svm.best_params_}")

# RandomizedSearchCV con el espacio completo
from scipy.stats import uniform, loguniform

param_dist_svm = {
    'C': loguniform(0.001, 1000),  # Distribución logarítmica
    'gamma': ['scale', 'auto'] + list(loguniform(0.001, 100).rvs(10)),
    'kernel': ['rbf', 'poly', 'sigmoid']
}

print(f"\n🎲 RandomizedSearchCV:")
start_time = time()
random_svm = RandomizedSearchCV(
    SVC(random_state=42),
    param_distributions=param_dist_svm,
    n_iter=50,  # Solo 50 combinaciones aleatorias
    cv=3,
    n_jobs=-1,
    random_state=42
)
random_svm.fit(X_small_scaled, y_small)
random_time = time() - start_time

print(f"⏱️  Tiempo: {random_time:.2f}s")
print(f"🎯 Mejor score: {random_svm.best_score_:.4f}")
print(f"⚙️  Mejores parámetros: {random_svm.best_params_}")

# Comparación
plt.figure(figsize=(15, 8))

# Comparar tiempos y scores
plt.subplot(2, 3, 1)
methods = ['GridSearchCV', 'RandomizedSearchCV']
times = [grid_time, random_time]
scores = [grid_svm.best_score_, random_svm.best_score_]

x = np.arange(len(methods))
plt.bar(x, times, alpha=0.7, color=['blue', 'orange'])
plt.xlabel('Método')
plt.ylabel('Tiempo (segundos)')
plt.title('Tiempo de Ejecución')
plt.xticks(x, methods)

for i, v in enumerate(times):
    plt.text(i, v + 0.1, f'{v:.1f}s', ha='center')

plt.subplot(2, 3, 2)
plt.bar(x, scores, alpha=0.7, color=['blue', 'orange'])
plt.xlabel('Método')
plt.ylabel('Score')
plt.title('Mejor Score Obtenido')
plt.xticks(x, methods)

for i, v in enumerate(scores):
    plt.text(i, v + 0.001, f'{v:.4f}', ha='center')

# Eficiencia (score por segundo)
plt.subplot(2, 3, 3)
efficiency = [s/t for s, t in zip(scores, times)]
plt.bar(x, efficiency, alpha=0.7, color=['blue', 'orange'])
plt.xlabel('Método')
plt.ylabel('Score / Segundo')
plt.title('Eficiencia')
plt.xticks(x, methods)

for i, v in enumerate(efficiency):
    plt.text(i, v + 0.0001, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.show()

# Recomendaciones
print(f"\n💡 RECOMENDACIONES:")
print(f"🔍 GridSearchCV:")
print(f"  ✅ Busca exhaustivamente en todo el espacio")
print(f"  ✅ Garantiza encontrar el óptimo en el grid definido")
print(f"  ❌ Puede ser muy lento con muchos parámetros")
print(f"  💡 Mejor para: Pocos parámetros, búsqueda final")

print(f"\n🎲 RandomizedSearchCV:")
print(f"  ✅ Mucho más rápido")
print(f"  ✅ Puede explorar valores continuos")
print(f"  ✅ Bueno para exploración inicial")
print(f"  ❌ No garantiza encontrar el óptimo global")
print(f"  💡 Mejor para: Muchos parámetros, búsqueda exploratoria")
```

### Pipeline con GridSearchCV

```python
# GridSearchCV con Pipeline completo
print(f"\n🔧 GRIDSEARCHCV CON PIPELINE COMPLETO")
print("=" * 60)

# Crear pipeline que incluye preprocesamiento y modelo
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

# Pipeline complejo
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest()),
    ('dimensionality_reduction', PCA()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Grid de parámetros para el pipeline completo
param_grid_pipeline = {
    # Parámetros del selector de características
    'feature_selection__k': [10, 15, 20],

    # Parámetros de PCA
    'dimensionality_reduction__n_components': [5, 10, 15],

    # Parámetros del clasificador
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5, None],
    'classifier__min_samples_split': [2, 5]
}

print(f"🔗 Pipeline: {' -> '.join([step[0] for step in pipeline.steps])}")

# Calcular combinaciones
total_pipeline = 1
for param, values in param_grid_pipeline.items():
    total_pipeline *= len(values)
    print(f"  {param}: {values}")

print(f"\n🔢 Total de combinaciones: {total_pipeline}")

# Ejecutar GridSearchCV con pipeline
print(f"\n⏳ Ejecutando GridSearchCV con Pipeline...")
start_time = time()

grid_pipeline = GridSearchCV(
    pipeline,
    param_grid_pipeline,
    cv=3,  # Reducir CV para que sea más rápido
    scoring='accuracy',
    n_jobs=-1
)

grid_pipeline.fit(X, y)
end_time = time()

print(f"\n🎯 RESULTADOS DEL PIPELINE:")
print(f"⏱️  Tiempo: {end_time - start_time:.2f} segundos")
print(f"🏆 Mejor score: {grid_pipeline.best_score_:.4f}")
print(f"⚙️  Mejores parámetros:")
for param, value in grid_pipeline.best_params_.items():
    print(f"    {param}: {value}")

# Analizar la importancia de cada etapa
print(f"\n🔍 ANÁLISIS DEL PIPELINE ÓPTIMO:")
best_pipeline = grid_pipeline.best_estimator_

print(f"📊 Características seleccionadas: {best_pipeline.named_steps['feature_selection'].k}")
print(f"🎯 Componentes PCA: {best_pipeline.named_steps['dimensionality_reduction'].n_components}")

# Ver características más importantes seleccionadas
feature_selector = best_pipeline.named_steps['feature_selection']
selected_features = feature_selector.get_support()
feature_scores = feature_selector.scores_

print(f"📈 Top 5 características más importantes:")
top_features_idx = np.argsort(feature_scores)[-5:][::-1]
for i, idx in enumerate(top_features_idx, 1):
    print(f"  {i}. Característica {idx}: Score = {feature_scores[idx]:.3f}")

# Visualizar el impacto de cada componente
plt.figure(figsize=(15, 10))

# Gráfico 1: Impacto del número de características
plt.subplot(2, 3, 1)
pipeline_results = pd.DataFrame(grid_pipeline.cv_results_)
k_values = sorted(pipeline_results['param_feature_selection__k'].unique())
k_scores = [pipeline_results[pipeline_results['param_feature_selection__k'] == k]['mean_test_score'].mean()
           for k in k_values]

plt.plot(k_values, k_scores, 'o-', color='blue', linewidth=2)
plt.xlabel('Número de Características (k)')
plt.ylabel('Score Promedio')
plt.title('Impacto de la Selección de Características')
plt.grid(True, alpha=0.3)

# Gráfico 2: Impacto de componentes PCA
plt.subplot(2, 3, 2)
pca_values = sorted(pipeline_results['param_dimensionality_reduction__n_components'].unique())
pca_scores = [pipeline_results[pipeline_results['param_dimensionality_reduction__n_components'] == n]['mean_test_score'].mean()
             for n in pca_values]

plt.plot(pca_values, pca_scores, 'o-', color='red', linewidth=2)
plt.xlabel('Componentes PCA')
plt.ylabel('Score Promedio')
plt.title('Impacto de la Reducción Dimensional')
plt.grid(True, alpha=0.3)

# Gráfico 3: Heatmap de combinaciones
plt.subplot(2, 3, 3)
pivot_pipeline = pipeline_results.groupby(['param_feature_selection__k',
                                         'param_dimensionality_reduction__n_components'])['mean_test_score'].mean().unstack()
sns.heatmap(pivot_pipeline, annot=True, fmt='.3f', cmap='viridis')
plt.title('Score: Características vs PCA')
plt.xlabel('Componentes PCA')
plt.ylabel('Características (k)')

plt.tight_layout()
plt.show()
```

### Consejos y Mejores Prácticas

```python
print(f"\n💡 MEJORES PRÁCTICAS PARA GRIDSEARCHCV")
print("=" * 60)

# 1. Búsqueda en dos etapas: Gruesa -> Fina
print(f"\n1️⃣  BÚSQUEDA EN DOS ETAPAS:")

# Etapa 1: Búsqueda gruesa
param_grid_coarse = {
    'n_estimators': [50, 200, 500],
    'max_depth': [3, 10, None],
    'learning_rate': [0.01, 0.1, 0.2]  # Ejemplo con GradientBoosting
}

# Etapa 2: Búsqueda fina (alrededor de los mejores valores)
param_grid_fine = {
    'n_estimators': [180, 200, 220],
    'max_depth': [8, 10, 12],
    'learning_rate': [0.08, 0.1, 0.12]
}

print("  Etapa 1 (Gruesa):", param_grid_coarse)
print("  Etapa 2 (Fina):", param_grid_fine)

# 2. Usando distribuciones para RandomizedSearchCV
print(f"\n2️⃣  DISTRIBUCIONES PARA RANDOMIZEDSEARCHCV:")
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 500),        # Enteros entre 50 y 500
    'max_depth': randint(3, 20),             # Enteros entre 3 y 20
    'learning_rate': uniform(0.01, 0.3),     # Flotantes entre 0.01 y 0.31
    'subsample': uniform(0.6, 0.4),          # Flotantes entre 0.6 y 1.0
}

print("  Distribuciones inteligentes:")
for param, dist in param_distributions.items():
    print(f"    {param}: {dist}")

# 3. Validación anidada para evaluación no sesgada
print(f"\n3️⃣  VALIDACIÓN ANIDADA (EVALUACIÓN NO SESGADA):")
from sklearn.model_selection import cross_val_score

# Problema: Si usamos los mismos datos para optimizar y evaluar, tenemos sesgo optimista
print("  ❌ INCORRECTO: Usar el mismo CV para optimizar Y evaluar")
print("     grid = GridSearchCV(model, params, cv=5)")
print("     score = grid.best_score_  # ¡SESGADO!")

print("  ✅ CORRECTO: Validación anidada")
print("     scores = cross_val_score(GridSearchCV(model, params, cv=3), X, y, cv=5)")
print("     # CV interno (3-fold) para optimizar, CV externo (5-fold) para evaluar")

# Ejemplo de validación anidada
nested_cv_scores = cross_val_score(
    GridSearchCV(RandomForestClassifier(random_state=42),
                {'n_estimators': [50, 100], 'max_depth': [3, 5]},
                cv=3),  # CV interno
    X[:200], y[:200],  # Dataset reducido para ejemplo
    cv=3  # CV externo
)

print(f"  📊 Scores de validación anidada: {nested_cv_scores}")
print(f"  📈 Score promedio no sesgado: {nested_cv_scores.mean():.4f} ± {nested_cv_scores.std():.4f}")

# 4. Monitoreo de recursos
print(f"\n4️⃣  MONITOREO DE RECURSOS:")
def estimate_gridsearch_time(param_grid, cv_folds=5, model_fit_time=1.0):
    """Estimar tiempo de GridSearchCV"""
    combinations = 1
    for values in param_grid.values():
        combinations *= len(values)

    total_fits = combinations * cv_folds
    estimated_time = total_fits * model_fit_time

    return combinations, total_fits, estimated_time

# Ejemplo
example_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

combs, fits, est_time = estimate_gridsearch_time(example_grid, cv_folds=5, model_fit_time=2.0)

print(f"  📊 Ejemplo de estimación:")
print(f"    Combinaciones: {combs}")
print(f"    Total de entrenamientos: {fits}")
print(f"    Tiempo estimado: {est_time:.0f} segundos ({est_time/60:.1f} minutos)")

# 5. Paralelización inteligente
print(f"\n5️⃣  PARALELIZACIÓN:")
print(f"  💻 n_jobs=-1: Usar todos los cores disponibles")
print(f"  ⚡ pre_dispatch='2*n_jobs': Controlar memoria")
print(f"  📊 verbose=1: Mostrar progreso")

example_grid_params = """
grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    n_jobs=-1,           # Todos los cores
    pre_dispatch='2*n_jobs',  # Control de memoria
    verbose=1,           # Mostrar progreso
    return_train_score=True   # Información adicional
)
"""
print(f"  📝 Ejemplo de configuración óptima:")
print(example_grid_params)

# Tabla resumen de cuándo usar cada método
print(f"\n📋 TABLA DE DECISIÓN:")
decision_table = """
┌─────────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ SITUACIÓN           │ MÉTODO          │ PARÁMETROS      │ TIEMPO          │
├─────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Pocos parámetros    │ GridSearchCV    │ < 100 combos    │ Minutos         │
│ (<3 parámetros)     │                 │                 │                 │
├─────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Muchos parámetros   │ RandomizedSearchCV │ n_iter=100   │ Horas -> Mins   │
│ (>3 parámetros)     │                 │                 │                 │
├─────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Exploración inicial │ RandomizedSearchCV │ n_iter=50    │ Exploración     │
│                     │                 │                 │ rápida          │
├─────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Refinamiento final  │ GridSearchCV    │ Rango pequeño   │ Precisión       │
│                     │                 │                 │                 │
├─────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Dataset grande      │ Validación      │ CV reducido     │ Compromiso      │
│                     │ Hold-out        │                 │                 │
└─────────────────────┴─────────────────┴─────────────────┴─────────────────┘
"""
print(decision_table)

print(f"\n✅ RESUMEN GRIDSEARCHCV:")
print(f"• Automatiza la búsqueda de hiperparámetros óptimos")
print(f"• GridSearchCV: exhaustivo pero lento")
print(f"• RandomizedSearchCV: rápido pero aproximado")
print(f"• Usar validación anidada para evaluación no sesgada")
print(f"• Combinar búsqueda gruesa + fina para eficiencia")
```

---

## Benchmarking y Mejores Prácticas

### ¿Qué es Benchmarking?

**Analogía deportiva:**
Imagina que eres un entrenador de fútbol y quieres saber qué tan bueno es tu equipo. No basta con que ganen un partido; necesitas:
1. Comparar con otros equipos (competencia)
2. Jugar en diferentes canchas (generalización)
3. Medir diferentes aspectos (velocidad, resistencia, técnica)
4. Hacer esto de forma justa y sistemática

**Benchmarking en ML** es exactamente esto: comparar modelos de forma sistemática y justa para decidir cuál es el mejor para tu problema específico.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Modelos a comparar
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

print("🏆 BENCHMARKING: COMPARACIÓN SISTEMÁTICA DE MODELOS")
print("=" * 70)

def create_benchmark_suite():
    """Crear suite de datasets para benchmarking"""
    datasets = {}

    # Dataset 1: Sintético balanceado
    X1, y1 = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                n_redundant=3, n_classes=2, random_state=42)
    datasets['Sintético_Balanceado'] = (X1, y1)

    # Dataset 2: Sintético desbalanceado
    X2, y2 = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                n_redundant=3, n_classes=2, weights=[0.9, 0.1], random_state=42)
    datasets['Sintético_Desbalanceado'] = (X2, y2)

    # Dataset 3: Breast Cancer (real)
    cancer = load_breast_cancer()
    datasets['Breast_Cancer'] = (cancer.data, cancer.target)

    # Dataset 4: Wine (multiclase)
    wine = load_wine()
    datasets['Wine'] = (wine.data, wine.target)

    return datasets

def create_model_suite():
    """Crear suite de modelos para benchmarking"""
    models = {
        'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM_RBF': SVC(kernel='rbf', random_state=42, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive_Bayes': GaussianNB(),
        'Decision_Tree': DecisionTreeClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }
    return models

# Crear suites de datos y modelos
datasets = create_benchmark_suite()
models = create_model_suite()

print(f"📊 Datasets disponibles: {list(datasets.keys())}")
print(f"🤖 Modelos a comparar: {list(models.keys())}")
```

### Benchmarking Completo con Múltiples Métricas

```python
def comprehensive_benchmark(datasets, models, cv_folds=5):
    """Realizar benchmarking completo con múltiples métricas"""

    results = {}
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr']

    print(f"\n🔄 Ejecutando benchmarking completo...")
    print(f"   Datasets: {len(datasets)}")
    print(f"   Modelos: {len(models)}")
    print(f"   Métricas: {len(metrics)}")
    print(f"   CV Folds: {cv_folds}")
    print(f"   Total evaluaciones: {len(datasets) * len(models) * len(metrics) * cv_folds}")

    for dataset_name, (X, y) in datasets.items():
        print(f"\n📊 Procesando dataset: {dataset_name}")

        # Estandarizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        results[dataset_name] = {}

        for model_name, model in models.items():
            try:
                # Crear pipeline
                if model_name.startswith('SVM'):
                    # SVM necesita datos escalados
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', model)
                    ])
                else:
                    pipeline = model

                # Validación cruzada con múltiples métricas
                cv_results = cross_validate(
                    pipeline, X, y,
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                    scoring=metrics,
                    return_train_score=True,
                    n_jobs=-1
                )

                # Almacenar resultados
                results[dataset_name][model_name] = {}
                for metric in metrics:
                    test_scores = cv_results[f'test_{metric}']
                    train_scores = cv_results[f'train_{metric}']

                    results[dataset_name][model_name][metric] = {
                        'test_mean': test_scores.mean(),
                        'test_std': test_scores.std(),
                        'train_mean': train_scores.mean(),
                        'train_std': train_scores.std(),
                        'overfitting': train_scores.mean() - test_scores.mean()
                    }

                print(f"   ✅ {model_name}: Completado")

            except Exception as e:
                print(f"   ❌ {model_name}: Error - {str(e)}")
                results[dataset_name][model_name] = None

    return results

# Ejecutar benchmarking
benchmark_results = comprehensive_benchmark(datasets, models, cv_folds=3)  # CV reducido para ejemplo

# Crear DataFrame para análisis
def results_to_dataframe(results):
    """Convertir resultados a DataFrame para fácil análisis"""
    rows = []
    for dataset_name, dataset_results in results.items():
        for model_name, model_results in dataset_results.items():
            if model_results is not None:
                for metric, scores in model_results.items():
                    row = {
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Metric': metric,
                        'Test_Mean': scores['test_mean'],
                        'Test_Std': scores['test_std'],
                        'Train_Mean': scores['train_mean'],
                        'Train_Std': scores['train_std'],
                        'Overfitting': scores['overfitting']
                    }
                    rows.append(row)
    return pd.DataFrame(rows)

df_results = results_to_dataframe(benchmark_results)
print(f"\n📋 DataFrame de resultados creado: {df_results.shape}")
print(df_results.head())
```

### Análisis y Visualización de Resultados

```python
# Análisis de resultados
print(f"\n📈 ANÁLISIS DE RESULTADOS DE BENCHMARKING")
print("=" * 60)

# 1. Mejor modelo por dataset y métrica
print(f"\n🏆 MEJORES MODELOS POR DATASET Y MÉTRICA:")
for dataset in df_results['Dataset'].unique():
    print(f"\n📊 {dataset}:")
    dataset_data = df_results[df_results['Dataset'] == dataset]

    for metric in ['accuracy', 'f1_macro', 'roc_auc_ovr']:
        metric_data = dataset_data[dataset_data['Metric'] == metric]
        if not metric_data.empty:
            best_model = metric_data.loc[metric_data['Test_Mean'].idxmax()]
            print(f"  {metric:12}: {best_model['Model']:20} ({best_model['Test_Mean']:.4f} ± {best_model['Test_Std']:.4f})")

# 2. Ranking global de modelos
print(f"\n🎖️  RANKING GLOBAL DE MODELOS (por accuracy):")
accuracy_data = df_results[df_results['Metric'] == 'accuracy']
global_ranking = accuracy_data.groupby('Model')['Test_Mean'].agg(['mean', 'std']).sort_values('mean', ascending=False)

for i, (model, scores) in enumerate(global_ranking.iterrows(), 1):
    print(f"{i:2d}. {model:20}: {scores['mean']:.4f} ± {scores['std']:.4f}")

# 3. Análisis de overfitting
print(f"\n⚠️  ANÁLISIS DE OVERFITTING:")
overfitting_analysis = df_results[df_results['Metric'] == 'accuracy'].groupby('Model')['Overfitting'].agg(['mean', 'std']).sort_values('mean', ascending=False)

print("Modelos más propensos al overfitting:")
for model, scores in overfitting_analysis.head().iterrows():
    status = "🔴" if scores['mean'] > 0.05 else "🟡" if scores['mean'] > 0.02 else "🟢"
    print(f"  {status} {model:20}: {scores['mean']:+.4f} ± {scores['std']:.4f}")

# Visualizaciones comprehensivas
fig = plt.figure(figsize=(20, 16))

# 1. Heatmap de rendimiento por dataset y modelo
plt.subplot(3, 3, 1)
accuracy_pivot = df_results[df_results['Metric'] == 'accuracy'].pivot(index='Model', columns='Dataset', values='Test_Mean')
sns.heatmap(accuracy_pivot, annot=True, fmt='.3f', cmap='RdYlGn', cbar_kws={'label': 'Accuracy'})
plt.title('Accuracy por Modelo y Dataset')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# 2. Box plot de distribución de scores
plt.subplot(3, 3, 2)
accuracy_data = df_results[df_results['Metric'] == 'accuracy']
sns.boxplot(data=accuracy_data, y='Model', x='Test_Mean', orient='h')
plt.title('Distribución de Accuracy por Modelo')
plt.xlabel('Accuracy Score')

# 3. Análisis de overfitting
plt.subplot(3, 3, 3)
overfitting_data = df_results[df_results['Metric'] == 'accuracy']
plt.scatter(overfitting_data['Train_Mean'], overfitting_data['Test_Mean'],
           c=overfitting_data['Overfitting'], cmap='RdYlBu_r', alpha=0.7, s=60)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('Train Score')
plt.ylabel('Test Score')
plt.title('Train vs Test Performance')
plt.colorbar(label='Overfitting Gap')

# 4. Radar chart comparando métricas del mejor modelo
plt.subplot(3, 3, 4, projection='polar')
best_model_overall = global_ranking.index[0]
model_data = df_results[(df_results['Model'] == best_model_overall) &
                       (df_results['Dataset'] == 'Breast_Cancer')]

metrics_for_radar = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr']
if not model_data.empty:
    values = []
    for metric in metrics_for_radar:
        metric_data = model_data[model_data['Metric'] == metric]
        if not metric_data.empty:
            values.append(metric_data['Test_Mean'].iloc[0])
        else:
            values.append(0)

    angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False)
    values += values[:1]  # Cerrar el polígono
    angles = np.concatenate((angles, [angles[0]]))

    ax = plt.gca()
    ax.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax.fill(angles, values, alpha=0.25, color='blue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_for_radar)
    ax.set_ylim(0, 1)
    plt.title(f'Radar: {best_model_overall}\n(Breast Cancer)', pad=20)

# 5. Estabilidad de modelos (std vs mean)
plt.subplot(3, 3, 5)
stability_data = df_results[df_results['Metric'] == 'accuracy']
scatter = plt.scatter(stability_data['Test_Mean'], stability_data['Test_Std'],
                     c=stability_data['Overfitting'], cmap='RdYlBu_r', alpha=0.7, s=60)
plt.xlabel('Mean Accuracy')
plt.ylabel('Standard Deviation')
plt.title('Estabilidad de Modelos')
plt.colorbar(scatter, label='Overfitting')

# 6. Comparación de métricas para un dataset específico
plt.subplot(3, 3, 6)
breast_cancer_data = df_results[df_results['Dataset'] == 'Breast_Cancer']
metrics_comparison = breast_cancer_data.pivot(index='Model', columns='Metric', values='Test_Mean')
metrics_comparison = metrics_comparison[['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']]
metrics_comparison.plot(kind='bar', ax=plt.gca(), width=0.8)
plt.title('Métricas - Breast Cancer Dataset')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 7. Tiempo vs Rendimiento (simulado)
plt.subplot(3, 3, 7)
# Simular tiempos de entrenamiento (en la práctica, se medirían)
model_complexity = {
    'Naive_Bayes': 0.1, 'Logistic_Regression': 0.3, 'KNN': 0.2,
    'Decision_Tree': 0.5, 'Random_Forest': 2.0, 'Gradient_Boosting': 3.0,
    'AdaBoost': 1.5, 'SVM_RBF': 2.5
}

accuracy_vs_time = []
for model in global_ranking.index:
    if model in model_complexity:
        accuracy_vs_time.append({
            'Model': model,
            'Accuracy': global_ranking.loc[model, 'mean'],
            'Time': model_complexity[model]
        })

df_time = pd.DataFrame(accuracy_vs_time)
plt.scatter(df_time['Time'], df_time['Accuracy'], s=100, alpha=0.7)
for i, row in df_time.iterrows():
    plt.annotate(row['Model'], (row['Time'], row['Accuracy']), xytext=(5, 5),
                textcoords='offset points', fontsize=8)
plt.xlabel('Tiempo de Entrenamiento (relativo)')
plt.ylabel('Accuracy Promedio')
plt.title('Tiempo vs Rendimiento')

plt.tight_layout()
plt.show()
```

### Interpretación de Modelos y Feature Importance

```python
# Análisis de interpretabilidad de modelos
print(f"\n🔍 ANÁLISIS DE INTERPRETABILIDAD")
print("=" * 50)

def analyze_feature_importance(X, y, feature_names=None):
    """Analizar importancia de características con diferentes modelos"""

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

    # Modelos que proporcionan feature importance
    interpretable_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    plt.figure(figsize=(15, 10))

    for i, (name, model) in enumerate(interpretable_models.items(), 1):
        # Entrenar modelo
        model.fit(X, y)

        # Obtener importancia de características
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]

        plt.subplot(2, 2, i)
        plt.bar(range(min(10, len(importance))), importance[indices[:10]])
        plt.title(f'Feature Importance - {name}')
        plt.xlabel('Características')
        plt.ylabel('Importancia')
        plt.xticks(range(min(10, len(importance))),
                  [feature_names[i] for i in indices[:10]], rotation=45)

    # Comparación de importancias
    plt.subplot(2, 2, 4)
    importance_comparison = {}
    for name, model in interpretable_models.items():
        model.fit(X, y)
        importance_comparison[name] = model.feature_importances_

    df_importance = pd.DataFrame(importance_comparison, index=feature_names)
    top_features = df_importance.sum(axis=1).nlargest(10).index

    df_importance.loc[top_features].plot(kind='bar', ax=plt.gca())
    plt.title('Comparación de Feature Importance')
    plt.xlabel('Características')
    plt.ylabel('Importancia')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    return importance_comparison

# Analizar importancia de características en Breast Cancer
breast_cancer = load_breast_cancer()
X_bc, y_bc = breast_cancer.data, breast_cancer.target

print(f"📊 Analizando importancia de características - Breast Cancer Dataset")
importance_results = analyze_feature_importance(X_bc, y_bc, breast_cancer.feature_names)

# Mostrar top características
print(f"\n🏆 TOP 5 CARACTERÍSTICAS MÁS IMPORTANTES (promedio):")
avg_importance = {}
for feature_idx, feature_name in enumerate(breast_cancer.feature_names):
    avg_imp = np.mean([imp[feature_idx] for imp in importance_results.values()])
    avg_importance[feature_name] = avg_imp

top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
for i, (feature, importance) in enumerate(top_features, 1):
    print(f"{i}. {feature}: {importance:.4f}")
```

### Consejos y Mejores Prácticas para Benchmarking

```python
print(f"\n💡 MEJORES PRÁCTICAS PARA BENCHMARKING")
print("=" * 60)

# 1. Protocolo de benchmarking
print(f"\n1️⃣  PROTOCOLO DE BENCHMARKING:")
protocol_steps = [
    "Definir el problema claramente",
    "Seleccionar datasets representativos",
    "Elegir métricas apropiadas para el problema",
    "Usar validación cruzada estratificada",
    "Estandarizar preprocesamiento",
    "Fijar semillas aleatorias para reproducibilidad",
    "Medir tanto rendimiento como tiempo",
    "Analizar overfitting y estabilidad",
    "Documentar todos los resultados"
]

for i, step in enumerate(protocol_steps, 1):
    print(f"   {i}. {step}")

# 2. Métricas según el tipo de problema
print(f"\n2️⃣  SELECCIÓN DE MÉTRICAS:")
metrics_guide = {
    "Clasificación Balanceada": ["accuracy", "f1_score", "roc_auc"],
    "Clasificación Desbalanceada": ["precision", "recall", "f1_score", "average_precision"],
    "Multiclase": ["accuracy", "f1_macro", "f1_weighted"],
    "Regresión": ["r2_score", "mean_squared_error", "mean_absolute_error"],
    "Ranking": ["ndcg", "map", "precision_at_k"]
}

for problem_type, metrics in metrics_guide.items():
    print(f"   {problem_type:25}: {', '.join(metrics)}")

# 3. Consideraciones estadísticas
print(f"\n3️⃣  CONSIDERACIONES ESTADÍSTICAS:")
statistical_considerations = [
    "Usar tests estadísticos para comparar modelos",
    "Reportar intervalos de confianza",
    "Considerar significancia práctica vs estadística",
    "Usar múltiples seeds para robustez",
    "Aplicar corrección de Bonferroni para múltiples comparaciones"
]

for consideration in statistical_considerations:
    print(f"   • {consideration}")

# 4. Test estadístico de comparación
print(f"\n4️⃣  EJEMPLO DE TEST ESTADÍSTICO:")

def statistical_comparison(results_model1, results_model2, alpha=0.05):
    """Comparar dos modelos estadísticamente usando t-test pareado"""
    from scipy import stats

    # T-test pareado
    t_stat, p_value = stats.ttest_rel(results_model1, results_model2)

    print(f"   T-statistic: {t_stat:.4f}")
    print(f"   P-value: {p_value:.4f}")

    if p_value < alpha:
        winner = "Modelo 1" if np.mean(results_model1) > np.mean(results_model2) else "Modelo 2"
        print(f"   ✅ Diferencia significativa (p < {alpha}). Ganador: {winner}")
    else:
        print(f"   ❌ No hay diferencia significativa (p >= {alpha})")

    return t_stat, p_value

# Ejemplo con datos simulados
np.random.seed(42)
model1_scores = np.random.normal(0.85, 0.03, 10)  # Modelo 1
model2_scores = np.random.normal(0.82, 0.04, 10)  # Modelo 2

print(f"Comparando Random Forest vs SVM:")
print(f"   Random Forest: {model1_scores.mean():.4f} ± {model1_scores.std():.4f}")
print(f"   SVM: {model2_scores.mean():.4f} ± {model2_scores.std():.4f}")

statistical_comparison(model1_scores, model2_scores)

# 5. Matriz de decisión para selección de modelos
print(f"\n5️⃣  MATRIZ DE DECISIÓN PARA SELECCIÓN:")

decision_matrix = """
┌──────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ CRITERIO         │ MUY IMPORTANTE  │ IMPORTANTE      │ MODERADO        │ MENOS IMPORTANTE│
├──────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Accuracy         │ Deep Learning   │ Random Forest   │ SVM             │ Naive Bayes     │
│                  │ Ensemble        │ Gradient Boost  │ Logistic Reg    │                 │
├──────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Velocidad        │ Naive Bayes     │ Logistic Reg    │ Decision Tree   │ SVM             │
│                  │ Linear Models   │ KNN (fast)      │ Random Forest   │ Deep Learning   │
├──────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Interpretabilidad│ Decision Tree   │ Linear Models   │ Random Forest   │ SVM             │
│                  │ Linear Reg      │ Naive Bayes     │ (importances)   │ Deep Learning   │
├──────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Datos Pequeños   │ Naive Bayes     │ Logistic Reg    │ SVM             │ Deep Learning   │
│ (<1000 samples)  │ KNN             │ Decision Tree   │                 │ Ensemble        │
├──────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Datos Grandes    │ Deep Learning   │ Logistic Reg    │ Random Forest   │ KNN             │
│ (>100k samples)  │ Linear Models   │ Gradient Boost  │                 │ Naive Bayes     │
└──────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘
"""

print(decision_matrix)

# 6. Checklist de benchmarking
print(f"\n6️⃣  CHECKLIST DE BENCHMARKING:")
checklist = [
    "☐ Definir métricas de éxito",
    "☐ Preparar múltiples datasets de prueba",
    "☐ Establecer baseline simple",
    "☐ Implementar validación cruzada",
    "☐ Fijar semillas aleatorias",
    "☐ Medir tiempo de entrenamiento e inferencia",
    "☐ Evaluar overfitting",
    "☐ Analizar errores cualitativamente",
    "☐ Probar con datos nuevos/externos",
    "☐ Documentar resultados y decisiones",
    "☐ Considerar restricciones de producción",
    "☐ Validar con stakeholders"
]

for item in checklist:
    print(f"   {item}")

print(f"\n✅ RESUMEN FINAL:")
print(f"• El benchmarking sistemático es crucial para la selección de modelos")
print(f"• Usar múltiples datasets y métricas para evaluación robusta")
print(f"• Considerar no solo accuracy, sino también tiempo e interpretabilidad")
print(f"• Aplicar tests estadísticos para comparaciones válidas")
print(f"• Documentar todo el proceso para reproducibilidad")
print(f"• El 'mejor' modelo depende del contexto específico del problema")

print(f"\n🎉 ¡FELICITACIONES!")
print(f"Has completado la guía completa de Machine Learning.")
print(f"Ahora tienes las herramientas para:")
print(f"• Entender y implementar modelos supervisados")
print(f"• Evaluar modelos correctamente con múltiples métricas")
print(f"• Usar validación cruzada para estimaciones robustas")
print(f"• Optimizar hiperparámetros con GridSearchCV")
print(f"• Realizar benchmarking profesional de modelos")
print(f"")
print(f"💡 ¡El siguiente paso es practicar con tus propios datos!")
```