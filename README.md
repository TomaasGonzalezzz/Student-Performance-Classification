# Student Performance Classification
### Clasificación y validación por escuela con NumPy

Predecir el rendimiento estudiantil exige tener en cuenta el desbalance de clases y las diferencias entre escuelas. En este trabajo comparo modelos de clasificación y analizo cómo cambia su evaluación al separar los datos por escuela o por período.

**Tomás González** · Machine Learning, UdeSA · 2026.

![Evaluación binaria del modelo final](assets/evaluacion-binaria-test.png)

[Notebook](notebook/Entrega_TP2.ipynb) · [Consigna](docs/consigna.pdf)

## Qué implementé

Regresión logística binaria y multiclase, LDA, árboles y Random Forest con NumPy. El análisis incluye validación cruzada por escuela, búsqueda de hiperparámetros, ponderación de costos, undersampling, duplicación y SMOTE dentro de cada fold de entrenamiento.

El conjunto original tiene 5.058 registros de desarrollo y 795 de test, correspondientes a ocho escuelas y seis semestres.

| Resultado guardado en test | Valor |
| --- | ---: |
| Accuracy binaria | 0,833 |
| F1 binario · «Aprobado» como positivo | 0,896 |
| Recall de «Insuficiente» | 0,989 |
| Accuracy multiclase | 0,692 |
| Macro F1 multiclase | 0,654 |

El modelo binario detecta 88 de 89 casos «Insuficiente», pero también señala a 132 aprobados. Esa diferencia entre detección y falsas alarmas es parte de la interpretación del resultado. Las cifras provienen de la entrega original.

## Abrir el trabajo

Con un entorno virtual activo:

~~~sh
python -m pip install -r requirements.txt
cd notebook
python -m jupyterlab
~~~

El notebook registra Python 3.12.2. Se necesita NumPy 2.0 o posterior por el uso de np.trapezoid. La ejecución completa recalcula búsquedas y guarda tablas; las salidas existentes permiten leer el trabajo sin entrenar de nuevo.

El código está en **src/**, los datos en **data/** y las tablas de experimentos en **outputs/tables/**.

[Entorno, estructura y verificaciones](DOCUMENTACION.md).
