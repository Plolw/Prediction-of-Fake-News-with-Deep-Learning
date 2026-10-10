# Detección de Fake News mediante Deep Learning

*(Basado en el proyecto académico: "Detecció de Fake News mitjançant Deep Learning")*

> 📄 **Consulta el informe técnico completo en:** [`project_report.pdf`](project_report.pdf)

## Objetivo del Proyecto
El objetivo principal de este trabajo es desarrollar un sistema automatizado para clasificar noticias (tanto el titular como el cuerpo del texto) como verdaderas o falsas (Fake News). Para ello, se aplican técnicas de Procesamiento de Lenguaje Natural (NLP) y diversas arquitecturas de redes neuronales profundas (Deep Learning) y modelos clásicos, evaluando su eficacia y la forma en que cada modelo "entiende" el lenguaje.

## Conjunto de Datos y Preprocesamiento
* **Origen:** Dataset "Fake News Classification Dataset" obtenido de Kaggle, centrado en noticias y política de los Estados Unidos (en inglés).
* **Tamaño:** Tras eliminar valores nulos y filtrar anomalías (noticias menores a 3 palabras o mayores a 4000), el corpus consta de **24.273 noticias**.
* **Preprocesamiento (NLP):**
  * Normalización a minúsculas y limpieza con expresiones regulares (Regex) para eliminar URLs y caracteres especiales.
  * Filtrado de *stop-words* y eliminación de palabras "trampa" (ej. "Reuters" que aparecía mayoritariamente en noticias reales).
  * Tokenización y vectorización limitando el vocabulario a las 10.000 palabras más frecuentes, ajustando la longitud de cada texto a 1.500 palabras (Padding).

## Modelos Implementados

### 1. Multi-Layer Perceptron (MLP) con Embeddings entrenados desde cero
* Arquitectura basada en una capa de Embeddings para la representación vectorial de palabras, seguida de capas Densas.
* Se aplicaron técnicas de regularización (Weight Decay L2 y Early Stopping) para mitigar el sobreajuste (overfitting).
* **Rendimiento:** Accuracy del **95%**. El modelo aprendió eficientemente a identificar palabras clave, actuando más como un detector de vocabulario que de contexto.

### 2. Long Short-Term Memory (LSTM) Bidireccional
* Se implementó una arquitectura LSTM para intentar capturar la dependencia temporal y el orden estructural de las palabras en el texto.
* Se utilizaron **Embeddings pre-entrenados (GloVe de Stanford)**, aplicando Fine-Tuning en la fase final del entrenamiento.
* **Rendimiento:** Accuracy del **96%**. A pesar del mayor coste computacional, el análisis de inferencia demostró que la mejora en la comprensión estructural fue marginal respecto a modelos más simples.

### 3. Regresión Logística con TF-IDF (Modelo Clásico)
* Como base de comparación, se implementó un modelo lineal de clasificación sobre vectores TF-IDF.
* **Rendimiento:** Logró un impresionante **96%** de Accuracy con un coste computacional ínfimo. Demostró que gran parte del problema radica en la distribución de vocabulario sensacionalista vs. profesional.

## Modelo Final: Ensemble (Soft Voting)
Dado que la Regresión Logística se fija en palabras individuales y la LSTM aporta (en teoría) cierto entendimiento contextual, el modelo final es un ensamble ponderado (Soft Voting) de ambos:
* **Fórmula:** $P = 0.3 \cdot P_{LSTM} + 0.7 \cdot P_{RegLog}$
* **Rendimiento Final:** Accuracy del **97%**, con F1-Score del 0.97 para ambas clases, consolidándose como el modelo más robusto y generalizable del estudio.

## Autor
Proyecto de Aprendizaje Automático II realizado en la Universitat Autònoma de Barcelona (UAB) por:
* Pol Ortiz Escobar