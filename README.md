# Student Performance Classification
### Clasificación y validación por escuela con NumPy

Predecir qué estudiantes van a tener un rendimiento insuficiente parece un problema de clasificación estándar, pero tiene dos trampas: las clases están muy desbalanceadas y las escuelas son muy distintas entre sí. En este trabajo implemento los modelos desde cero y analizo cómo cambia la evaluación al tener en cuenta las dos cosas.

**Tomás González** · Machine Learning, UdeSA · 2026

[Notebook](notebook/Entrega_TP2.ipynb) · [Consigna](docs/consigna.pdf)

**Resultado:** el modelo final detecta 88 de los 89 estudiantes en riesgo del test, evaluado sobre un flujo elegido con validación por escuela.

![Tasa de aprobación por escuela y semestre](assets/aprobacion-por-escuela.png)

## Recorrido

### 1. Explorar los datos

5.058 registros de desarrollo y 795 de test, de ocho escuelas y seis semestres. La exploración mostró enseguida el problema: en algunas escuelas aprueba casi todo el mundo y en otras, cerca de la mitad.

![Distribución del objetivo por escuela](assets/objetivo-por-escuela.png)

Por eso comparé tres formas de separar los datos. La partición al azar da resultados optimistas, la temporal queda en un punto intermedio y la validación **por escuela**, sobre escuelas que el modelo no vio, es el escenario más duro. Es la que usé para elegir los modelos.

### 2. Implementar los modelos

Con NumPy implementé regresión logística binaria y multiclase, LDA, árboles de decisión y Random Forest, junto con la búsqueda de hiperparámetros y las métricas.

### 3. Tratar el desbalance

Comparé distintas estrategias para que el modelo no ignore la clase minoritaria: ponderación de costos, undersampling, duplicación y SMOTE, siempre aplicadas dentro de cada fold de entrenamiento para no filtrar información.

![Matrices de confusión de las distintas estrategias](assets/matrices-confusion.png)

Mirando solo las métricas globales, no rebalancear parece lo mejor. Pero para detectar a quienes están en riesgo, ponderar los costos recupera muchos más casos, así que elegí una regresión logística L2 con esa estrategia. En el caso multiclase, la regresión logística fue la opción más sólida.

### 4. Evaluar en test

![Evaluación binaria del modelo final](assets/evaluacion-binaria-test.png)

| Resultado en test | Valor |
| --- | ---: |
| Accuracy binaria | 0,833 |
| F1 binario · «Aprobado» como positivo | 0,896 |
| Recall de «Insuficiente» | 0,989 |
| Accuracy multiclase | 0,692 |
| Macro F1 multiclase | 0,654 |

El modelo binario detecta 88 de los 89 casos «Insuficiente», a costa de marcar también a 132 estudiantes que aprobaron. Para un sistema de alerta temprana, ese equilibrio entre detectar y generar falsas alarmas es la decisión central.

## Conclusión

El mejor modelo depende de la pregunta y de cómo se lo evalúa. La misma regresión logística parece excelente si se mezclan las escuelas y bastante más modesta frente a una escuela nueva. Y si el objetivo es detectar riesgo, conviene aceptar más falsas alarmas a cambio de no dejar pasar casos. También estimé que el Random Forest necesitaría unos 9.118 datos más para ganar cerca de 1 % de accuracy.

## Abrir el trabajo

Con un entorno virtual activo:

~~~sh
python -m pip install -r requirements.txt
cd notebook
python -m jupyterlab
~~~

Se necesita NumPy 2.0 o posterior. El notebook conserva sus salidas, así que se puede leer sin reentrenar. El código está en **src/**, los datos en **data/** y las tablas de experimentos en **outputs/tables/**.

[Entorno, estructura y verificaciones](DOCUMENTACION.md)
