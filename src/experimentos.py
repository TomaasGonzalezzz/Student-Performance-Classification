from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from configuracion import ORDEN_OBJETIVO_MULTICLASE
from metricas import reporte_clasificacion_binaria, reporte_multiclase_one_vs_all
from modelos import (
    AnalisisDiscriminanteLinealCovarianzaCompartida,
    ClasificadorRandomForestEntropia,
    RegresionLogisticaBinariaL2,
    RegresionLogisticaMulticlaseL2,
)
from preprocesamiento import ajustar_preprocesador_modelo, aplicar_preprocesador_modelo
from utilidades import guardar_dataframe
from validation import construir_indices_group_kfold

REBALANCING_TECHNIQUES_ORDER = [
    "no_rebalancing",
    "undersampling",
    "oversampling_duplicate",
    "oversampling_smote",
    "cost_reweighting",
]

#Solo para darle mejores nombres en los graficos

REBALANCING_TECHNIQUE_LABELS = {
    "no_rebalancing": "No rebalancing",
    "undersampling": "Undersampling",
    "oversampling_duplicate": "Duplicate oversampling",
    "oversampling_smote": "SMOTE oversampling",
    "cost_reweighting": "Cost re-weighting",
}

def iterar_folds(datos: pd.DataFrame, folds):
    for fold_id, (idx_train, idx_valid) in enumerate(folds, start=1):
        yield fold_id, datos.iloc[idx_train].copy(), datos.iloc[idx_valid].copy(), idx_valid

def preparar_datos_fold(
    datos_entrenamiento: pd.DataFrame,
    datos_validation: pd.DataFrame,
    columna_objetivo: str,
) -> dict:
    estado_preprocesador = ajustar_preprocesador_modelo(datos_entrenamiento)
    X_train = aplicar_preprocesador_modelo(datos_entrenamiento, estado_preprocesador)
    X_validation = aplicar_preprocesador_modelo(datos_validation, estado_preprocesador)
    return {
        "preprocessor_state": estado_preprocesador,
        "X_train": X_train,
        "X_validation": X_validation,
        "y_train": datos_entrenamiento[columna_objetivo].to_numpy(dtype=int),
        "y_validation": datos_validation[columna_objetivo].to_numpy(dtype=int),
    }

def seleccionar_mejor_lambda(
    resumen: pd.DataFrame,
    f1_col: str,
    pr_auc_col: str,
    roc_auc_col: str,
    tolerance: float = 1e-4,
) -> float:
    best_f1 = float(resumen[f1_col].max())
    candidatos = resumen[resumen[f1_col] >= best_f1 - tolerance].copy()
    mejor_fila = candidatos.sort_values(
        [f1_col, pr_auc_col, roc_auc_col, "lambda"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return float(mejor_fila["lambda"])

# Rebalancing binario

def resumir_conteos_binarios(y) -> dict:
    y = np.asarray(y, dtype=int)
    count_0 = int(np.sum(y == 0))
    count_1 = int(np.sum(y == 1))
    minority_class = 0 if count_0 <= count_1 else 1
    majority_class = 1 - minority_class
    return {
        "count_0": count_0,
        "count_1": count_1,
        "minority_class": int(minority_class),
        "majority_class": int(majority_class),
    }

def calcular_cost_weights(y) -> tuple[np.ndarray, float, int]:
    y = np.asarray(y, dtype=int)
    conteos = resumir_conteos_binarios(y)
    minority_class = conteos["minority_class"]
    minority_count = conteos[f"count_{minority_class}"]
    majority_count = conteos[f"count_{conteos['majority_class']}"]
    if minority_count == 0:
        return np.ones(len(y), dtype=float), 1.0, minority_class
    pesos = np.ones(len(y), dtype=float)
    pesos[y == minority_class] = majority_count / minority_count
    return pesos, float(majority_count / minority_count), int(minority_class)

def separar_indices_clases(y: np.ndarray) -> tuple[dict, dict]:
    conteos = resumir_conteos_binarios(y)
    minority_class = conteos["minority_class"]
    majority_class = conteos["majority_class"]
    return conteos, {
        "minority": np.where(y == minority_class)[0],
        "majority": np.where(y == majority_class)[0],
    }

def aplicar_undersampling(X, y, semilla: int = 42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    _, indices = separar_indices_clases(y)
    if len(indices["minority"]) == 0 or len(indices["majority"]) == 0:
        return X.copy(), y.copy()
    rng = np.random.default_rng(semilla)
    sampled_majority = rng.choice(indices["majority"], size=len(indices["minority"]), replace=False)
    selected = np.sort(np.concatenate([indices["minority"], sampled_majority]))
    return X[selected], y[selected]

def aplicar_duplicate_oversampling(X, y, semilla: int = 42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    _, indices = separar_indices_clases(y)
    n_extra = len(indices["majority"]) - len(indices["minority"])
    if n_extra <= 0 or len(indices["minority"]) == 0:
        return X.copy(), y.copy()
    rng = np.random.default_rng(semilla)
    duplicated = rng.choice(indices["minority"], size=n_extra, replace=True)
    return np.vstack([X, X[duplicated]]), np.concatenate([y, y[duplicated]])

def aplicar_smote_oversampling(X, y, k_neighbors: int = 5, semilla: int = 42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    _, indices = separar_indices_clases(y)
    n_extra = len(indices["majority"]) - len(indices["minority"])
    if n_extra <= 0 or len(indices["minority"]) == 0:
        return X.copy(), y.copy()
    if len(indices["minority"]) < 2:
        return aplicar_duplicate_oversampling(X, y, semilla=semilla)

    X_minority = X[indices["minority"]]
    k_neighbors = min(int(k_neighbors), len(X_minority) - 1)
    if k_neighbors < 1:
        return aplicar_duplicate_oversampling(X, y, semilla=semilla)

    diff = X_minority[:, None, :] - X_minority[None, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(distances, np.inf)
    neighbor_idx = np.argsort(distances, axis=1)[:, :k_neighbors]

    rng = np.random.default_rng(semilla)
    synthetic_rows = []
    for _ in range(n_extra):
        anchor_pos = int(rng.integers(len(X_minority)))
        neighbor_pos = int(rng.choice(neighbor_idx[anchor_pos]))
        gap = float(rng.random())
        anchor = X_minority[anchor_pos]
        neighbor = X_minority[neighbor_pos]
        synthetic_rows.append(anchor + gap * (neighbor - anchor))

    synthetic_X = np.asarray(synthetic_rows, dtype=float)
    synthetic_y = np.full(n_extra, resumir_conteos_binarios(y)["minority_class"], dtype=int)
    return np.vstack([X, synthetic_X]), np.concatenate([y, synthetic_y])

def aplicar_rebalancing_technique(
    X,
    y,
    method: str,
    semilla: int = 42,
    smote_k_neighbors: int = 5,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    original_counts = resumir_conteos_binarios(y)

    if method == "cost_reweighting":
        sample_weight, weight_factor, _ = calcular_cost_weights(y)
        X_fit, y_fit = X.copy(), y.copy()
    else:
        methods_map = {
            "no_rebalancing": lambda: (X.copy(), y.copy()),
            "undersampling": lambda: aplicar_undersampling(X, y, semilla=semilla),
            "oversampling_duplicate": lambda: aplicar_duplicate_oversampling(X, y, semilla=semilla),
            "oversampling_smote": lambda: aplicar_smote_oversampling(
                X, y, k_neighbors=smote_k_neighbors, semilla=semilla
            ),
        }
        if method not in methods_map:
            raise ValueError(f"Unknown method: {method}")
        X_fit, y_fit = methods_map[method]()
        sample_weight, weight_factor = None, 1.0

    fitted_counts = resumir_conteos_binarios(y_fit)
    return {
        "X_fit": X_fit,
        "y_fit": y_fit,
        "sample_weight": sample_weight,
        "weight_factor": float(weight_factor),
        "summary": {
            "technique": method,
            "technique_label": REBALANCING_TECHNIQUE_LABELS[method],
            "original_count_0": original_counts["count_0"],
            "original_count_1": original_counts["count_1"],
            "fitted_count_0": fitted_counts["count_0"],
            "fitted_count_1": fitted_counts["count_1"],
            "minority_class": original_counts["minority_class"],
            "majority_class": original_counts["majority_class"],
            "minority_weight_factor": float(weight_factor),
        },
    }

def entrenar_y_evaluar_binario(
    datos_entrenamiento: pd.DataFrame,
    datos_validation: pd.DataFrame,
    lambda_: float,
    columna_objetivo: str,
    parametros_modelo: dict | None = None,
    method: str = "no_rebalancing",
    semilla: int = 42,
    smote_k_neighbors: int = 5,
) -> dict:
    datos = preparar_datos_fold(datos_entrenamiento, datos_validation, columna_objetivo)
    rebalance_result = aplicar_rebalancing_technique(
        datos["X_train"].to_numpy(),
        datos["y_train"],
        method=method,
        semilla=semilla,
        smote_k_neighbors=smote_k_neighbors,
    )

    model = RegresionLogisticaBinariaL2(lambda_=lambda_, **(parametros_modelo or {}))
    model.fit(
        rebalance_result["X_fit"],
        rebalance_result["y_fit"],
        pesos_muestra=rebalance_result["sample_weight"],
    )

    validation_scores = model.predict_proba(datos["X_validation"].to_numpy())
    reporte = reporte_clasificacion_binaria(datos["y_validation"], validation_scores)
    return {
        "y_validation": datos["y_validation"],
        "validation_scores": validation_scores,
        "metrics": reporte["metrics"],
        "confusion_matrix": reporte["confusion_matrix"],
        "roc_curve": reporte["roc_curve"],
        "pr_curve": reporte["pr_curve"],
        "coefficients": pd.DataFrame({"feature": datos["X_train"].columns, "coefficient": model.coef_}),
    }

def resumir_metricas_binarias(resultados_folds: pd.DataFrame, columnas_grupo) -> pd.DataFrame:
    return (
        resultados_folds.groupby(columnas_grupo, as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            mean_f1=("f1_score", "mean"),
            std_f1=("f1_score", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
        )
        .fillna(0.0)
    )

def evaluar_grilla_binaria(
    datos: pd.DataFrame | None,
    folds,
    grilla_lambda,
    columna_objetivo: str,
    parametros_modelo: dict | None,
    fila_base: dict,
    method: str = "no_rebalancing",
    semilla_base: int = 42,
    smote_k_neighbors: int = 5,
    holdout: tuple[pd.DataFrame, pd.DataFrame] | None = None,
):
    rows = []
    for lambda_id, lambda_ in enumerate(grilla_lambda, start=1):
        if holdout is not None:
            fold_iter = [(None, holdout[0], holdout[1], None)]
        else:
            fold_iter = iterar_folds(datos, folds)

        for fold, df_train, df_valid, _ in fold_iter:
            evaluacion = entrenar_y_evaluar_binario(
                datos_entrenamiento=df_train,
                datos_validation=df_valid,
                lambda_=lambda_,
                columna_objetivo=columna_objetivo,
                parametros_modelo=parametros_modelo,
                method=method,
                semilla=semilla_base + 1000 * lambda_id + (fold or 0),
                smote_k_neighbors=smote_k_neighbors,
            )
            extra = {**fila_base, "lambda": float(lambda_)}
            if fold is not None:
                extra["fold"] = fold
            rows.append({**evaluacion["metrics"], **extra})

    resultados_folds = None if holdout is not None else pd.DataFrame(rows)
    if holdout is None:
        resumen = resumir_metricas_binarias(resultados_folds, list(fila_base) + ["lambda"])
        resumen = resumen.sort_values("lambda").reset_index(drop=True)
        best_lambda = seleccionar_mejor_lambda(
            resumen, f1_col="mean_f1", pr_auc_col="mean_pr_auc", roc_auc_col="mean_roc_auc"
        )
    else:
        resumen = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
        best_lambda = seleccionar_mejor_lambda(
            resumen, f1_col="f1_score", pr_auc_col="pr_auc", roc_auc_col="roc_auc"
        )
    return resultados_folds, resumen, best_lambda

def obtener_coeficientes_desde_folds(
    datos: pd.DataFrame,
    folds,
    lambda_: float,
    columna_objetivo: str = "rendimiento_binario_num",
    parametros_modelo=None,
    strategy_name: str = "strategy",
):
    filas = []
    for fold, df_train, df_valid, _ in iterar_folds(datos, folds):
        coeficientes = entrenar_y_evaluar_binario(
            datos_entrenamiento=df_train,
            datos_validation=df_valid,
            lambda_=lambda_,
            columna_objetivo=columna_objetivo,
            parametros_modelo=parametros_modelo,
        )["coefficients"].copy()
        coeficientes["strategy"] = strategy_name
        coeficientes["fold"] = fold
        filas.append(coeficientes)
    return pd.concat(filas, ignore_index=True)

def resumir_estabilidad_coeficientes(coeficientes: pd.DataFrame) -> pd.DataFrame:
    return (
        coeficientes.groupby(["strategy", "feature"], as_index=False)
        .agg(
            coef_mean=("coefficient", "mean"),
            coef_std=("coefficient", "std"),
            coef_abs_mean=("coefficient", lambda valores: np.mean(np.abs(valores))),
        )
        .fillna(0.0)
    )

def comparar_estabilidad_coeficientes(
    resumen_random_tabla: pd.DataFrame,
    resumen_group_tabla: pd.DataFrame,
) -> pd.DataFrame:
    merged = resumen_random_tabla.merge(
        resumen_group_tabla,
        on="feature",
        suffixes=("_random", "_group"),
    )
    merged["std_gap_group_minus_random"] = merged["coef_std_group"] - merged["coef_std_random"]
    merged["std_ratio_group_over_random"] = np.where(
        merged["coef_std_random"] > 0,
        merged["coef_std_group"] / merged["coef_std_random"],
        np.nan,
    )
    return merged.sort_values(
        ["std_gap_group_minus_random", "coef_abs_mean_group"],
        ascending=[False, False],
    ).reset_index(drop=True)

def evaluar_rebalancing_out_of_fold(
    datos: pd.DataFrame,
    folds,
    method: str,
    lambda_: float,
    columna_objetivo: str = "rendimiento_binario_num",
    parametros_modelo=None,
    semilla_base: int = 42,
    smote_k_neighbors: int = 5,
):
    y_true = datos[columna_objetivo].to_numpy(dtype=int)
    oof_scores = np.zeros(len(datos), dtype=float)

    for fold, df_train, df_valid, idx_valid in iterar_folds(datos, folds):
        evaluacion = entrenar_y_evaluar_binario(
            datos_entrenamiento=df_train,
            datos_validation=df_valid,
            lambda_=lambda_,
            columna_objetivo=columna_objetivo,
            parametros_modelo=parametros_modelo,
            method=method,
            semilla=semilla_base + 1000 + fold,
            smote_k_neighbors=smote_k_neighbors,
        )
        oof_scores[idx_valid] = evaluacion["validation_scores"]

    reporte = reporte_clasificacion_binaria(y_true, oof_scores)
    metrics_row = {
        **reporte["metrics"],
        "technique": method,
        "technique_label": REBALANCING_TECHNIQUE_LABELS[method],
        "lambda": float(lambda_),
    }
    return {
        "metrics": metrics_row,
        "confusion_matrix": reporte["confusion_matrix"],
        "roc_curve": reporte["roc_curve"],
        "pr_curve": reporte["pr_curve"],
        "positive_rate": float(np.mean(y_true)),
    }

def resumir_rebalancing_example(
    datos_entrenamiento: pd.DataFrame,
    methods,
    columna_objetivo: str = "rendimiento_binario_num",
    smote_k_neighbors: int = 5,
):
    datos = preparar_datos_fold(datos_entrenamiento, datos_entrenamiento, columna_objetivo)
    filas = []
    for method_id, method in enumerate(methods, start=1):
        filas.append(
            aplicar_rebalancing_technique(
                datos["X_train"].to_numpy(),
                datos["y_train"],
                method=method,
                semilla=42 + method_id,
                smote_k_neighbors=smote_k_neighbors,
            )["summary"]
        )
    return pd.DataFrame(filas)

# Multiclase

def construir_modelo_multiclase(nombre_modelo: str, parametros_modelo=None):
    parametros_modelo = dict(parametros_modelo or {})
    if nombre_modelo == "lda":
        return AnalisisDiscriminanteLinealCovarianzaCompartida(**parametros_modelo)
    if nombre_modelo == "multiclass_logistic":
        return RegresionLogisticaMulticlaseL2(**parametros_modelo)
    if nombre_modelo == "random_forest":
        if "random_state" in parametros_modelo and "semilla" not in parametros_modelo:
            parametros_modelo["semilla"] = parametros_modelo.pop("random_state")
        else:
            parametros_modelo.pop("random_state", None)
        return ClasificadorRandomForestEntropia(**parametros_modelo)
    raise ValueError(f"Modelo desconocido: {nombre_modelo}")

def formatear_etiqueta_configuracion(nombre_modelo: str, parametros_modelo: dict) -> str:
    if nombre_modelo == "lda":
        return f"LDA | ridge={parametros_modelo.get('covariance_ridge', 1e-4)}"
    if nombre_modelo == "multiclass_logistic":
        return f"Logistica | lambda={parametros_modelo.get('lambda_', 1.0)}"
    if nombre_modelo == "random_forest":
        return (
            f"RF | arboles={parametros_modelo.get('n_estimators')} | "
            f"prof={parametros_modelo.get('max_depth')} | "
            f"hoja={parametros_modelo.get('min_samples_leaf')} | "
            f"vars={parametros_modelo.get('max_features')}"
        )
    return nombre_modelo

def tabla_importancias(model, columnas) -> pd.DataFrame | None:
    if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
        return pd.DataFrame({"feature": columnas, "importance": model.feature_importances_})
    return None

def entrenar_pipeline_multiclase(
    datos_entrenamiento: pd.DataFrame,
    datos_validation: pd.DataFrame,
    nombre_modelo: str,
    parametros_modelo=None,
    columna_objetivo: str = "rendimiento_ordinal",
):
    datos = preparar_datos_fold(datos_entrenamiento, datos_validation, columna_objetivo)
    model = construir_modelo_multiclase(nombre_modelo, parametros_modelo=parametros_modelo)
    model.fit(datos["X_train"].to_numpy(), datos["y_train"])
    validation_scores = model.predict_proba(datos["X_validation"].to_numpy())
    reporte = reporte_multiclase_one_vs_all(
        datos["y_validation"],
        validation_scores,
        etiquetas_clases=ORDEN_OBJETIVO_MULTICLASE,
    )
    return {
        "validation_scores": validation_scores,
        "metrics": reporte["metrics"],
        "per_class_metrics": reporte["per_class_metrics"],
        "confusion_matrix": reporte["confusion_matrix"],
        "roc_curves": reporte["roc_curves"],
        "pr_curves": reporte["pr_curves"],
        "positive_rates": reporte["positive_rates"],
        "feature_importances": tabla_importancias(model, datos["X_train"].columns),
    }

def seleccionar_mejor_configuracion_multiclase(
    resumen: pd.DataFrame,
    tolerance: float = 1e-4,
) -> dict:
    best_f1 = float(resumen["mean_macro_f1"].max())
    candidatos = resumen[resumen["mean_macro_f1"] >= best_f1 - tolerance].copy()
    mejor_fila = candidatos.sort_values(
        ["mean_macro_f1", "mean_overall_accuracy", "mean_macro_roc_auc"],
        ascending=[False, False, False],
    ).iloc[0]
    return mejor_fila.to_dict()

def resumen_configuraciones_multiclase(resultados_folds: pd.DataFrame) -> pd.DataFrame:
    return (
        resultados_folds.groupby(["model_name", "config_id", "config_label"], as_index=False)
        .agg(
            mean_overall_accuracy=("overall_accuracy", "mean"),
            std_overall_accuracy=("overall_accuracy", "std"),
            mean_macro_accuracy=("macro_accuracy", "mean"),
            mean_macro_precision=("macro_precision", "mean"),
            mean_macro_recall=("macro_recall", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_macro_roc_auc=("macro_roc_auc", "mean"),
            mean_macro_pr_auc=("macro_pr_auc", "mean"),
        )
        .fillna(0.0)
    )

def cross_validation_configuraciones_multiclase(
    datos: pd.DataFrame,
    folds,
    nombre_modelo: str,
    grilla_configuraciones,
    columna_objetivo: str = "rendimiento_ordinal",
):
    fold_rows, metadata = [], []
    for config_id, parametros_modelo in enumerate(grilla_configuraciones, start=1):
        config_label = formatear_etiqueta_configuracion(nombre_modelo, parametros_modelo)
        metadata.append({"config_id": config_id, "config_label": config_label, **parametros_modelo})
        for fold, df_train, df_valid, _ in iterar_folds(datos, folds):
            evaluacion = entrenar_pipeline_multiclase(
                datos_entrenamiento=df_train,
                datos_validation=df_valid,
                nombre_modelo=nombre_modelo,
                parametros_modelo=parametros_modelo,
                columna_objetivo=columna_objetivo,
            )
            fold_rows.append(
                {
                    **evaluacion["metrics"],
                    "model_name": nombre_modelo,
                    "config_id": config_id,
                    "config_label": config_label,
                    "fold": fold,
                    **parametros_modelo,
                }
            )

    resultados_folds = pd.DataFrame(fold_rows)
    resumen = resumen_configuraciones_multiclase(resultados_folds)
    resumen = resumen.merge(pd.DataFrame(metadata).drop_duplicates("config_id"), on=["config_id", "config_label"], how="left")
    resumen = resumen.sort_values(
        ["mean_macro_f1", "mean_overall_accuracy", "mean_macro_roc_auc"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return resultados_folds, resumen, seleccionar_mejor_configuracion_multiclase(resumen)

def evaluar_multiclase_out_of_fold(
    datos: pd.DataFrame,
    folds,
    nombre_modelo: str,
    parametros_modelo=None,
    columna_objetivo: str = "rendimiento_ordinal",
    etiqueta_modelo: str | None = None,
):
    etiqueta_modelo = etiqueta_modelo or nombre_modelo
    y_true = datos[columna_objetivo].to_numpy(dtype=int)
    n_classes = len(ORDEN_OBJETIVO_MULTICLASE)
    oof_scores = np.zeros((len(datos), n_classes), dtype=float)
    feature_importance_rows = []

    for fold, df_train, df_valid, idx_valid in iterar_folds(datos, folds):
        evaluacion = entrenar_pipeline_multiclase(
            datos_entrenamiento=df_train,
            datos_validation=df_valid,
            nombre_modelo=nombre_modelo,
            parametros_modelo=parametros_modelo,
            columna_objetivo=columna_objetivo,
        )
        oof_scores[idx_valid] = evaluacion["validation_scores"]

        if evaluacion["feature_importances"] is not None:
            importance = evaluacion["feature_importances"].copy()
            importance["model"] = etiqueta_modelo
            importance["fold"] = fold
            feature_importance_rows.append(importance)

    reporte = reporte_multiclase_one_vs_all(y_true, oof_scores, etiquetas_clases=ORDEN_OBJETIVO_MULTICLASE)
    per_class_report = reporte["per_class_metrics"].copy()
    per_class_report["model"] = etiqueta_modelo
    return {
        "model": etiqueta_modelo,
        "model_name": nombre_modelo,
        "model_params": parametros_modelo or {},
        "metrics": {**reporte["metrics"], "model": etiqueta_modelo},
        "per_class_metrics": per_class_report,
        "confusion_matrix": reporte["confusion_matrix"],
        "roc_curves": reporte["roc_curves"],
        "pr_curves": reporte["pr_curves"],
        "positive_rates": reporte["positive_rates"],
        "feature_importances": (
            pd.concat(feature_importance_rows, ignore_index=True) if feature_importance_rows else None
        ),
    }

def resumir_importancias_features(importancia_features: pd.DataFrame) -> pd.DataFrame:
    return (
        importancia_features.groupby(["model", "feature"], as_index=False)
        .agg(mean_importance=("importance", "mean"), std_importance=("importance", "std"))
        .fillna(0.0)
        .sort_values(["model", "mean_importance"], ascending=[True, False])
        .reset_index(drop=True)
    )

# Punto 5 y curva de aprendizaje

def construir_tabla_seleccion_punto5(eleccion_binaria: dict, eleccion_multiclase: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task": "Binario",
                "model": eleccion_binaria["model_label"],
                "rebalancing": eleccion_binaria["rebalancing_label"],
                "selection_strategy": eleccion_binaria["validation_strategy"],
                "final_fit_data": "todo dev",
                "selected_lambda": eleccion_binaria.get("lambda"),
                "main_reason": eleccion_binaria["main_reason"],
            },
            {
                "task": "Multiclase",
                "model": eleccion_multiclase["model_label"],
                "rebalancing": eleccion_multiclase.get("rebalancing_label", "No aplica"),
                "selection_strategy": eleccion_multiclase["validation_strategy"],
                "final_fit_data": "todo dev",
                "selected_lambda": eleccion_multiclase.get("lambda"),
                "main_reason": eleccion_multiclase["main_reason"],
            },
        ]
    )

def resumir_metricas_riesgo_binario(matriz_confusion: pd.DataFrame) -> dict:
    tp = int(matriz_confusion.loc["real_0", "pred_0"])
    fn = int(matriz_confusion.loc["real_0", "pred_1"])
    fp = int(matriz_confusion.loc["real_1", "pred_0"])
    tn = int(matriz_confusion.loc["real_1", "pred_1"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "insuf_precision": precision,
        "insuf_recall": recall,
        "insuf_f1": f1_score,
        "insuf_tp": tp,
        "insuf_fp": fp,
        "insuf_fn": fn,
        "insuf_tn": tn,
    }

def evaluar_pipeline_final_binario(
    datos_desarrollo: pd.DataFrame,
    datos_prueba: pd.DataFrame,
    lambda_: float,
    method: str = "no_rebalancing",
    parametros_modelo=None,
    columna_objetivo: str = "rendimiento_binario_num",
    etiqueta_estrategia: str = "Temporal split",
):
    evaluacion = entrenar_y_evaluar_binario(
        datos_entrenamiento=datos_desarrollo,
        datos_validation=datos_prueba,
        lambda_=lambda_,
        columna_objetivo=columna_objetivo,
        parametros_modelo=parametros_modelo,
        method=method,
    )
    metrics_row = {
        **evaluacion["metrics"],
        **resumir_metricas_riesgo_binario(evaluacion["confusion_matrix"]),
        "task": "Binario",
        "model": "Logistica binaria L2",
        "rebalancing": method,
        "rebalancing_label": REBALANCING_TECHNIQUE_LABELS[method],
        "validation_strategy": etiqueta_estrategia,
        "lambda": float(lambda_),
    }
    return {
        "metrics": metrics_row,
        "metrics_df": pd.DataFrame([metrics_row]),
        "confusion_matrix": evaluacion["confusion_matrix"],
        "roc_curve": evaluacion["roc_curve"],
        "pr_curve": evaluacion["pr_curve"],
        "positive_rate": float(datos_prueba[columna_objetivo].mean()),
    }

def evaluar_pipeline_final_multiclase(
    datos_desarrollo: pd.DataFrame,
    datos_prueba: pd.DataFrame,
    nombre_modelo: str,
    parametros_modelo: dict,
    etiqueta_modelo: str,
    validation_strategy: str,
    columna_objetivo: str = "rendimiento_ordinal",
):
    evaluacion = entrenar_pipeline_multiclase(
        datos_entrenamiento=datos_desarrollo,
        datos_validation=datos_prueba,
        nombre_modelo=nombre_modelo,
        parametros_modelo=parametros_modelo,
        columna_objetivo=columna_objetivo,
    )
    metrics_row = {
        **evaluacion["metrics"],
        "task": "Multiclase",
        "model": etiqueta_modelo,
        "validation_strategy": validation_strategy,
    }
    per_class_metrics_tabla = evaluacion["per_class_metrics"].copy()
    per_class_metrics_tabla["model"] = etiqueta_modelo
    return {
        "metrics": metrics_row,
        "metrics_df": pd.DataFrame([metrics_row]),
        "per_class_metrics": per_class_metrics_tabla,
        "confusion_matrix": evaluacion["confusion_matrix"],
        "roc_curves": evaluacion["roc_curves"],
        "pr_curves": evaluacion["pr_curves"],
        "positive_rates": evaluacion["positive_rates"],
    }

def muestrear_fraccion_dentro_grupos(
    datos: pd.DataFrame,
    fraction: float,
    columna_grupo: str = "escuela",
    min_rows_per_group: int = 40,
    semilla: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    sampled_parts = []
    for _, datos_grupo in datos.groupby(columna_grupo, observed=True):
        n_group = len(datos_grupo)
        n_sample = min(n_group, max(min_rows_per_group, int(round(n_group * fraction))))
        chosen_idx = rng.choice(datos_grupo.index.to_numpy(), size=n_sample, replace=False)
        sampled_parts.append(datos.loc[np.sort(chosen_idx)])
    return pd.concat(sampled_parts, ignore_index=True)

def correr_curva_aprendizaje_rf_multiclase(
    datos_desarrollo: pd.DataFrame,
    parametros_rf: dict,
    fractions,
    repeats: int = 2,
    columna_objetivo: str = "rendimiento_ordinal",
    columna_grupo: str = "escuela",
    semilla: int = 42,
) -> dict:
    fractions = [float(valor) for valor in fractions]
    measurement_rows = []
    for fraction_id, fraction in enumerate(fractions, start=1):
        for repeat in range(repeats):
            sample_seed = semilla + 100 * fraction_id + repeat
            sampled_tabla = muestrear_fraccion_dentro_grupos(
                datos_desarrollo,
                fraction=fraction,
                columna_grupo=columna_grupo,
                semilla=sample_seed,
            )
            folds = construir_indices_group_kfold(sampled_tabla[columna_grupo], n_folds=4)
            _, resumen, _ = cross_validation_configuraciones_multiclase(
                sampled_tabla.reset_index(drop=True),
                folds,
                nombre_modelo="random_forest",
                grilla_configuraciones=[parametros_rf],
                columna_objetivo=columna_objetivo,
            )
            measurement_rows.append(
                {
                    **resumen.iloc[0].to_dict(),
                    "fraction": fraction,
                    "repeat": repeat + 1,
                    "n_samples": len(sampled_tabla),
                }
            )

    mediciones = pd.DataFrame(measurement_rows)
    resumen_curva = (
        mediciones.groupby("fraction", as_index=False)
        .agg(
            n_samples=("n_samples", "mean"),
            mean_accuracy=("mean_overall_accuracy", "mean"),
            std_accuracy=("mean_overall_accuracy", "std"),
            mean_macro_f1=("mean_macro_f1", "mean"),
            std_macro_f1=("mean_macro_f1", "std"),
        )
        .fillna(0.0)
        .sort_values("n_samples")
        .reset_index(drop=True)
    )

    n_observaciones = resumen_curva["n_samples"].to_numpy(dtype=float)
    accuracy_media = resumen_curva["mean_accuracy"].to_numpy(dtype=float)
    x_ajuste = np.log(n_observaciones)
    slope, intercept = np.polyfit(x_ajuste, accuracy_media, deg=1)
    resumen_curva["fitted_accuracy"] = intercept + slope * np.log(n_observaciones)

    current_n = int(round(float(resumen_curva["n_samples"].max())))
    current_accuracy = float(resumen_curva.loc[resumen_curva["n_samples"].idxmax(), "mean_accuracy"])
    target_accuracy = current_accuracy + 0.01
    finite_solution = bool(slope > 0)
    if finite_solution:
        required_n = max(float(np.exp((target_accuracy - intercept) / slope)), float(current_n))
        additional_n = int(math.ceil(required_n - current_n))
    else:
        required_n = np.nan
        additional_n = np.nan

    fit_start_n = float(n_observaciones.min())
    curve_n_max = current_n * 1.12 if not finite_solution else max(current_n * 1.02, required_n * 1.05)
    curve_n = np.linspace(fit_start_n, curve_n_max, 200)
    curva_ajustada = pd.DataFrame(
        {"n_samples": curve_n, "predicted_accuracy": intercept + slope * np.log(curve_n)}
    )
    estimacion = pd.DataFrame(
        [
            {
                "current_n": current_n,
                "current_accuracy": current_accuracy,
                "target_accuracy": target_accuracy,
                "curve_intercept": intercept,
                "curve_slope": slope,
                "fit_method": "ajuste logaritmico sobre toda la curva",
                "fit_points": len(resumen_curva),
                "fit_start_n": fit_start_n,
                "required_n": required_n,
                "additional_n": additional_n,
                "finite_solution": finite_solution,
            }
        ]
    )
    return {
        "curve_summary_df": resumen_curva,
        "fitted_curve_df": curva_ajustada,
        "estimate_df": estimacion,
    }

def obtener_curva_aprendizaje_rf(
    datos_desarrollo: pd.DataFrame,
    parametros_rf: dict,
    fractions,
    summary_path,
    fitted_path,
    estimate_path,
    repeats: int = 2,
    columna_objetivo: str = "rendimiento_ordinal",
    columna_grupo: str = "escuela",
    semilla: int = 42,
) -> dict:
    summary_path = Path(summary_path)
    fitted_path = Path(fitted_path)
    estimate_path = Path(estimate_path)

    if all(path.exists() for path in [summary_path, fitted_path, estimate_path]):
        return {
            "curve_summary_df": pd.read_csv(summary_path),
            "fitted_curve_df": pd.read_csv(fitted_path),
            "estimate_df": pd.read_csv(estimate_path),
        }

    resultados = correr_curva_aprendizaje_rf_multiclase(
        datos_desarrollo=datos_desarrollo,
        parametros_rf=parametros_rf,
        fractions=fractions,
        repeats=repeats,
        columna_objetivo=columna_objetivo,
        columna_grupo=columna_grupo,
        semilla=semilla,
    )
    guardar_dataframe(resultados["curve_summary_df"], summary_path, index=False, habilitado=True)
    guardar_dataframe(resultados["fitted_curve_df"], fitted_path, index=False, habilitado=True)
    guardar_dataframe(resultados["estimate_df"], estimate_path, index=False, habilitado=True)
    return resultados
