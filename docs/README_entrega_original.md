# Gonzalez Tomas - TP2

Entrega del TP2 de `I302 - Aprendizaje Automatico y Aprendizaje Profundo`.

El notebook principal esta en `notebook/Entrega_TP2.ipynb`. El codigo auxiliar esta separado en `src/` para que el notebook quede mas limpio y se concentre en la presentacion de resultados, graficos y lecturas.

## Archivos principales

- `notebook/Entrega_TP2.ipynb`: notebook final de la entrega. Ejecuta el flujo completo del TP y muestra las tablas, graficos y comentarios de cada punto.
- `README.md`: resumen de la estructura del trabajo.

## Datos

- `data/raw/rendimiento_estudiantes_dev.csv`: base de desarrollo original provista por la catedra.
- `data/raw/rendimiento_estudiantes_test.csv`: base de test original provista por la catedra.
- `data/processed/rendimiento_estudiantes_dev_clean.csv`: base de desarrollo luego de la limpieza usada en el TP.
- `data/processed/rendimiento_estudiantes_test_clean.csv`: base de test luego de aplicar la misma limpieza.

## Codigo en `src/`

- `configuracion.py`: define rutas, semillas, columnas principales y parametros generales del trabajo.
- `datos.py`: carga los CSV originales y aplica la limpieza basica de las bases.
- `analisis_exploratorio.py`: arma resumenes para revisar faltantes, distribuciones, outliers, features por escuela y relaciones con la target.
- `preprocesamiento.py`: contiene imputacion, estandarizacion y armado de matrices para los modelos.
- `validation.py`: construye splits de train/validation, incluyendo Random split, Group split y Temporal split.
- `modelos.py`: implementa los modelos usados en el TP: regresion logistica binaria, regresion logistica multiclase, LDA, arbol de decision y random forest.
- `metricas.py`: calcula metricas, matrices de confusion y curvas ROC / precision-recall sin depender de librerias externas de Machine Learning.
- `experimentos.py`: entrena, valida y compara modelos para los distintos puntos del TP. Tambien incluye rebalancing y evaluacion final.
- `secciones_notebook.py`: conecta el codigo auxiliar con el notebook. Prepara los resultados que se muestran en cada punto.
- `visualizacion.py`: concentra las funciones de graficacion usadas en el notebook.
- `utilidades.py`: helpers chicos para crear carpetas y guardar salidas.

## Tablas generadas

- `outputs/tables/punto_3/`: resultados de validation y seleccion de modelos multiclase.
- `outputs/tables/punto_4/`: resultados de busqueda y evaluacion con rebalancing methods.
- `outputs/tables/punto_5/`: tablas usadas para la evaluacion final y la curva de aprendizaje del random forest.

Estas tablas quedan guardadas para no recalcular todo cada vez que se abre el notebook.

## Nota sobre graficos

El archivo `src/visualizacion.py` contiene funciones de graficacion y puede considerarse codigo asistido por IA, dentro de lo permitido por la consigna.
