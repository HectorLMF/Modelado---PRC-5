# Clasificador k-NN con Configuración Avanzada

Este proyecto implementa un **clasificador k-NN** altamente configurable. Permite seleccionar **métricas de distancia**, **pesado de vecinos**, **reglas de clasificación**, **normalización de datos**, y **almacenar experimentos para replicación posterior**. También se genera automáticamente un **informe detallado** con la configuración del experimento y la matriz de confusión.

## 🚀 Uso del programa

1. **Ejecutar `Main.java`** en IntelliJ IDEA o cualquier entorno compatible con Java.
2. **Seleccionar el dataset de entrada** (Ej.: `iris.csv`, `glass.csv`).
3. **Elegir la normalización**: `RAW`, `MIN-MAX`, `Z-SCORE`.
4. **Configurar el valor de `k`** para el algoritmo k-NN.
5. **Seleccionar la métrica de distancia** (`EuclideanDistance`, `ManhattanDistance`, `ChebyshevDistance`).
6. **Seleccionar la estrategia de pesado de vecinos** (`EqualWeighting`, `DistanceInverseWeighting`, `RankBasedWeighting`).
7. **Elegir el algoritmo de votación** (`MajorityVote`, `ThresholdVote`).
8. **Ingresar una nueva instancia para clasificación** (valores sin la etiqueta de clase).
9. **Ejecutar el experimento**, mostrando la **precisión y matriz de confusión**.
10. **Guardar los conjuntos de entrenamiento y prueba** en `datasets/` con nombres basados en la fecha y hora.
11. **Generar un informe del experimento** en `experiments_output/`, registrando:
    - Ruta del dataset original
    - Ruta de los conjuntos generados (`train.csv`, `test.csv`)
    - Parámetros del algoritmo (`k`, métrica, pesado, regla de clasificación)
    - Matriz de confusión y precisión

## 📂 Estructura del Código

El programa está compuesto por varias clases organizadas en paquetes.

### **Paquete `model` (Gestión de Datos)**  
Define la estructura de los datos utilizados en el clasificador.

- `Attribute`: Representa un atributo en el dataset con nombre, tipo y si es de clase.
- `Instance`: Representa una instancia de datos con valores asociados.
- `Dataset`: Almacena y maneja el conjunto de datos. Incluye normalización y extracción de estadísticas de atributos.
- `NormalizationMode`: Enumeración que define los tipos de normalización disponibles (`RAW`, `MIN_MAX`, `Z_SCORE`).

### **Paquete `classification` (Reglas de Clasificación)**  

- `ClassificationRule`: Interfaz para definir reglas de clasificación.
- `MajorityVote`: Usa mayoría simple para determinar la clase.
- `ThresholdVote`: Requiere un umbral mínimo de votos para aceptar una clase.

### **Paquete `distance` (Métricas de Distancia)**  

- `DistanceMetric`: Interfaz que define una métrica de distancia.
- `EuclideanDistance`: Implementa la distancia Euclidiana.
- `ManhattanDistance`: Implementa la distancia Manhattan.
- `ChebyshevDistance`: Implementa la distancia Chebyshev.

### **Paquete `weighting` (Pesado de Vecinos)**  

- `CaseWeightingStrategy`: Interfaz para estrategias de pesado de vecinos.
- `EqualWeighting`: Todos los vecinos tienen el mismo peso.
- `DistanceInverseWeighting`: Peso inversamente proporcional a la distancia.
- `RankBasedWeighting`: Asigna pesos decrecientes según el orden del vecino.

### **Paquete `knn` (Clasificador k-NN)**  

- `KNNClassifier`: Implementa el algoritmo k-NN con múltiples configuraciones.
  - `k`: Número de vecinos considerados.
  - `distanceMetric`: Métrica de distancia usada.
  - `classificationRule`: Algoritmo de votación.
  - `caseWeightingStrategy`: Estrategia de pesado de vecinos.
  - Métodos: `classify(instance)`, `setK(k)`, entre otros.

### **Paquete `experiment` (Gestión de Experimentos)**  

- `ExperimentManager`: Administra la generación y ejecución de experimentos.
  - `splitDatasetRatio(testRatio, random, seed)`: Divide el dataset en entrenamiento y prueba.
  - `saveSplit()`: Guarda los conjuntos en `datasets/` con nombres basados en la fecha y hora.
  - `runExperiment()`: Ejecuta el experimento y muestra resultados.
  - `generateExperimentReport()`: Genera un informe del experimento en `experiments_output/`.

## 🛠 Instalación y Ejecución  

1. **Descargar los datasets** (`glass.csv`, `iris.csv`) y guardarlos en `datasets/`.
2. **Compilar el código** usando:  
   ```bash
   javac -d bin $(find . -name "*.java")
3. **Ejecutar el programa** con:
  java -cp bin app.Main

4. Seguir las instrucciones en consola para configurar el clasificador.
5. Visualizar los resultados y el informe generado en experiments_output/.

