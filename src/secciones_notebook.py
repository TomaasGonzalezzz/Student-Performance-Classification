from __future__ import annotations

import pandas as pd

from analisis_exploratorio import (
    comparar_metricas_escala_nota_previa,
    resumir_escala_nota_previa,
    resumir_impacto_reescalado_nota_previa,
)
from configuracion import (
    CARPETA_DATOS_PROCESADOS,
    CARPETA_TABLAS_PUNTO_1,
)
from datos import cargar_dataset_crudo, limpiar_dataframe_estudiantes
from experimentos import (
    REBALANCING_TECHNIQUE_LABELS,
    REBALANCING_TECHNIQUES_ORDER,
    comparar_estabilidad_coeficientes,
    construir_tabla_seleccion_punto5,
    entrenar_y_evaluar_binario,
    evaluar_grilla_binaria,
    evaluar_multiclase_out_of_fold,
    evaluar_pipeline_final_binario,
    evaluar_pipeline_final_multiclase,
    evaluar_rebalancing_out_of_fold,
    obtener_coeficientes_desde_folds,
    obtener_curva_aprendizaje_rf,
    resumir_rebalancing_example,
    resumir_estabilidad_coeficientes,
    resumir_importancias_features,
    seleccionar_mejor_configuracion_multiclase,
    cross_validation_configuraciones_multiclase,
)
from preprocesamiento import ajustar_imputador_mediana_por_grupo, aplicar_imputador_mediana_por_grupo
from utilidades import guardar_dataframe
from validation import (
    construir_indices_random_kfold,
    construir_indices_kfold_estratificado,
    construir_indices_group_kfold,
    dividir_random_estratificado,
    dividir_group_holdout,
    dividir_holdout_temporal,
    elegir_group_holdout_schools,
)
from visualizacion import previsualizar_tabla

PARAMETROS_LOGISTICA_BASE = {
    "learning_rate": 0.05,
    "max_iter": 2500,
    "tolerance": 1e-6,
}

GRILLA_LAMBDA_BASE = [
    1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0,
]

GRILLA_RIDGE_LDA_BASE = [
    1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0,
]

FRACCIONES_CURVA_APRENDIZAJE_RF_BASE = [0.25, 0.4, 0.55, 0.7, 0.85, 1.0]

def configurar_random_forest(
    n_estimators: int,
    max_depth,
    min_samples_leaf: int = 5,
    max_features="sqrt",
    min_samples_split: int = 10,
    semilla: int = 42,
) -> dict:
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "semilla": semilla,
    }

def parametros_rf_desde_config(configuracion: dict) -> dict:
    profundidad = configuracion.get("max_depth")
    if pd.isna(profundidad):
        profundidad = None
    return {
        "n_estimators": int(configuracion["n_estimators"]),
        "max_depth": None if profundidad is None else int(profundidad),
        "min_samples_split": int(configuracion["min_samples_split"]),
        "min_samples_leaf": int(configuracion["min_samples_leaf"]),
        "max_features": configuracion["max_features"],
        "semilla": int(configuracion.get("semilla", configuracion.get("random_state", 42))),
    }

def preparar_datasets_entrega() -> dict:
    desarrollo_crudo = cargar_dataset_crudo("dev")
    prueba_crudo = cargar_dataset_crudo("test")
    desarrollo = limpiar_dataframe_estudiantes(desarrollo_crudo)
    prueba = limpiar_dataframe_estudiantes(prueba_crudo)

    estado_imputador = ajustar_imputador_mediana_por_grupo(desarrollo)
    desarrollo_modelo_base = aplicar_imputador_mediana_por_grupo(desarrollo, estado_imputador)
    prueba_modelo_base = aplicar_imputador_mediana_por_grupo(prueba, estado_imputador)

    guardar_dataframe(desarrollo_modelo_base, CARPETA_DATOS_PROCESADOS / "rendimiento_estudiantes_dev_clean.csv", index=False)
    guardar_dataframe(prueba_modelo_base, CARPETA_DATOS_PROCESADOS / "rendimiento_estudiantes_test_clean.csv", index=False)

    return {
        "dev_crudo": desarrollo_crudo,
        "test_crudo": prueba_crudo,
        "dev": desarrollo,
        "test": prueba,
        "dev_modelo_base": desarrollo_modelo_base,
        "test_modelo_base": prueba_modelo_base,
    }

def preparar_auditoria_nota_previa(desarrollo_crudo: pd.DataFrame, prueba_crudo: pd.DataFrame) -> dict:
    resumen_escala_tabla = resumir_escala_nota_previa(desarrollo_crudo)
    metricas_escala_tabla = comparar_metricas_escala_nota_previa(desarrollo_crudo)
    impacto_reescalado_tabla = pd.concat(
        [
            resumir_impacto_reescalado_nota_previa(desarrollo_crudo).assign(split="dev"),
            resumir_impacto_reescalado_nota_previa(prueba_crudo).assign(split="test"),
        ],
        ignore_index=True,
    )

    nota_previa_cruda = pd.to_numeric(desarrollo_crudo["nota_previa"], errors="coerce")
    nota_previa_valida = nota_previa_cruda[nota_previa_cruda.between(0, 10, inclusive="both")]
    nota_previa_sospechosa = nota_previa_cruda[(nota_previa_cruda > 10) & (nota_previa_cruda <= 100)]
    patron_formato_tabla = pd.DataFrame(
        [
            {
                "group": "valid_0_10",
                "n": int(nota_previa_valida.shape[0]),
                "share_with_decimals": (nota_previa_valida.mod(1) != 0).mean(),
                "share_integer": (nota_previa_valida.mod(1) == 0).mean(),
            },
            {
                "group": "suspect_10_100",
                "n": int(nota_previa_sospechosa.shape[0]),
                "share_with_decimals": (nota_previa_sospechosa.mod(1) != 0).mean(),
                "share_integer": (nota_previa_sospechosa.mod(1) == 0).mean(),
            },
        ]
    ).round(4)

    for tabla, ruta in [
        (resumen_escala_tabla, CARPETA_TABLAS_PUNTO_1 / "nota_previa_scale_summary.csv"),
        (metricas_escala_tabla, CARPETA_TABLAS_PUNTO_1 / "nota_previa_scale_metrics.csv"),
        (impacto_reescalado_tabla, CARPETA_TABLAS_PUNTO_1 / "nota_previa_rescaling_effect.csv"),
        (patron_formato_tabla, CARPETA_TABLAS_PUNTO_1 / "nota_previa_scale_format.csv"),
    ]:
        guardar_dataframe(tabla, ruta, index=False)

    return {
        "resumen_escala_df": resumen_escala_tabla,
        "metricas_escala_df": metricas_escala_tabla,
        "impacto_reescalado_df": impacto_reescalado_tabla,
        "patron_formato_df": patron_formato_tabla,
    }

def preparar_punto2(
    desarrollo: pd.DataFrame,
    columna_objetivo: str = "rendimiento_binario_num",
    grilla_lambda=None,
    parametros_modelo=None,
) -> dict:
    grilla_lambda = list(grilla_lambda or GRILLA_LAMBDA_BASE)
    parametros_modelo = dict(parametros_modelo or PARAMETROS_LOGISTICA_BASE)

    train_random, valid_random = dividir_random_estratificado(
        desarrollo,
        columna_objetivo=columna_objetivo,
        test_size=0.2,
        semilla=42,
    )
    holdout_schools, _ = elegir_group_holdout_schools(desarrollo, columna_objetivo=columna_objetivo)
    train_group, valid_group = dividir_group_holdout(desarrollo, holdout_schools)
    train_time, valid_time, semestre_validation = dividir_holdout_temporal(desarrollo)

    return {
        "grilla_lambda": grilla_lambda,
        "parametros_modelo": parametros_modelo,
        "columna_target": columna_objetivo,
        "train_random": train_random,
        "validation_random": valid_random,
        "holdout_schools": holdout_schools,
        "train_group": train_group,
        "validation_group": valid_group,
        "train_temporal": train_time,
        "validation_temporal": valid_time,
        "temporal_validation_semester": semestre_validation,
    }

def correr_punto2_binario(desarrollo: pd.DataFrame, setup: dict) -> dict:
    grilla_lambda = setup["grilla_lambda"]
    parametros_modelo = setup["parametros_modelo"]
    columna_objetivo = setup["columna_target"]

    definiciones = {
        "random": {
            "label": "Random split",
            "train": setup["train_random"],
            "valid": setup["validation_random"],
            "folds": construir_indices_kfold_estratificado(
                setup["train_random"][columna_objetivo].to_numpy(), n_folds=4, semilla=42
            ),
        },
        "group": {
            "label": "Group split",
            "train": setup["train_group"],
            "valid": setup["validation_group"],
            "folds": construir_indices_group_kfold(setup["train_group"]["escuela"], n_folds=3),
        },
        "temporal": {
            "label": "Temporal split",
            "train": setup["train_temporal"],
            "valid": setup["validation_temporal"],
            "folds": None,
        },
    }

    estrategias = {}
    for clave, spec in definiciones.items():
        if spec["folds"] is None:
            resultados_cv, ajuste, mejor_lambda = evaluar_grilla_binaria(
                datos=None,
                folds=None,
                grilla_lambda=grilla_lambda,
                columna_objetivo=columna_objetivo,
                parametros_modelo=parametros_modelo,
                fila_base={"strategy": spec["label"]},
                holdout=(spec["train"], spec["valid"]),
            )
        else:
            resultados_cv, ajuste, mejor_lambda = evaluar_grilla_binaria(
                datos=spec["train"].reset_index(drop=True),
                folds=spec["folds"],
                grilla_lambda=grilla_lambda,
                columna_objetivo=columna_objetivo,
                parametros_modelo=parametros_modelo,
                fila_base={"strategy": spec["label"]},
            )
        estrategias[clave] = {
            "label": spec["label"],
            "resultados_cv": resultados_cv,
            "ajuste": ajuste,
            "mejor_lambda": mejor_lambda,
            "evaluacion": entrenar_y_evaluar_binario(
                datos_entrenamiento=spec["train"],
                datos_validation=spec["valid"],
                lambda_=mejor_lambda,
                columna_objetivo=columna_objetivo,
                parametros_modelo=parametros_modelo,
            ),
        }

    best_lambda_tabla = pd.DataFrame(
        [{"strategy": v["label"], "best_lambda": v["mejor_lambda"]} for v in estrategias.values()]
    )
    metricas_estrategias_tabla = pd.DataFrame(
        [
            {"strategy": v["label"], "best_lambda": v["mejor_lambda"], **v["evaluacion"]["metrics"]}
            for v in estrategias.values()
        ]
    ).round(4)

    folds_group_reference = construir_indices_group_kfold(desarrollo["escuela"], n_folds=4)
    _, _, lambda_referencia = evaluar_grilla_binaria(
        datos=desarrollo.reset_index(drop=True),
        folds=folds_group_reference,
        grilla_lambda=grilla_lambda,
        columna_objetivo=columna_objetivo,
        parametros_modelo=parametros_modelo,
        fila_base={"strategy": "GroupKFold reference"},
    )

    coeficientes_random_tabla = obtener_coeficientes_desde_folds(
        desarrollo.reset_index(drop=True),
        folds=construir_indices_random_kfold(len(desarrollo), n_folds=5, semilla=42),
        lambda_=lambda_referencia,
        columna_objetivo=columna_objetivo,
        parametros_modelo=parametros_modelo,
        strategy_name="Random KFold",
    )
    coeficientes_group_tabla = obtener_coeficientes_desde_folds(
        desarrollo.reset_index(drop=True),
        folds=folds_group_reference,
        lambda_=lambda_referencia,
        columna_objetivo=columna_objetivo,
        parametros_modelo=parametros_modelo,
        strategy_name="GroupKFold",
    )
    distribucion_coeficientes_tabla = pd.concat([coeficientes_random_tabla, coeficientes_group_tabla], ignore_index=True)

    estabilidad_coeficientes_tabla = comparar_estabilidad_coeficientes(
        resumir_estabilidad_coeficientes(coeficientes_random_tabla).query("strategy == 'Random KFold'"),
        resumir_estabilidad_coeficientes(coeficientes_group_tabla).query("strategy == 'GroupKFold'"),
    )
    top_inestabilidad_tabla = estabilidad_coeficientes_tabla[
        ["feature", "coef_std_random", "coef_std_group", "std_gap_group_minus_random", "std_ratio_group_over_random"]
    ].head(15).round(4)
    return {
        "ajuste_random_df": estrategias["random"]["ajuste"],
        "ajuste_group_df": estrategias["group"]["ajuste"],
        "ajuste_temporal_df": estrategias["temporal"]["ajuste"],
        "mejores_lambda_df": best_lambda_tabla,
        "metricas_estrategias_df": metricas_estrategias_tabla,
        "evaluacion_random": estrategias["random"]["evaluacion"],
        "evaluacion_group": estrategias["group"]["evaluacion"],
        "evaluacion_temporal": estrategias["temporal"]["evaluacion"],
        "matriz_confusion_random_df": estrategias["random"]["evaluacion"]["confusion_matrix"],
        "matriz_confusion_group_df": estrategias["group"]["evaluacion"]["confusion_matrix"],
        "matriz_confusion_temporal_df": estrategias["temporal"]["evaluacion"]["confusion_matrix"],
        "tasa_positiva_random": estrategias["random"]["evaluacion"]["y_validation"].mean(),
        "tasa_positiva_group": estrategias["group"]["evaluacion"]["y_validation"].mean(),
        "tasa_positiva_temporal": estrategias["temporal"]["evaluacion"]["y_validation"].mean(),
        "distribucion_coeficientes_df": distribucion_coeficientes_tabla,
        "estabilidad_coeficientes_df": estabilidad_coeficientes_tabla,
        "features_inestables": top_inestabilidad_tabla["feature"].tolist(),
    }

def preparar_punto3(desarrollo: pd.DataFrame, n_folds: int = 4) -> dict:
    folds = construir_indices_group_kfold(desarrollo["escuela"], n_folds=n_folds)
    return {"folds_group_multiclase": folds}

def ajustar_modelos_punto3(
    desarrollo: pd.DataFrame,
    folds,
) -> dict:
    lda_ridge_grid = list(GRILLA_RIDGE_LDA_BASE)
    grilla_configuracion_lda = [{"covariance_ridge": valor} for valor in lda_ridge_grid]

    grilla_lambda_multiclase = list(GRILLA_LAMBDA_BASE)
    grilla_configuracion_logistica = [
        {"lambda_": valor, **PARAMETROS_LOGISTICA_BASE} for valor in grilla_lambda_multiclase
    ]

    grilla_configuracion_rf = [
        configurar_random_forest(30, 8),
        configurar_random_forest(40, 8),
        configurar_random_forest(30, 10),
        configurar_random_forest(40, 10),
        configurar_random_forest(30, 8, min_samples_leaf=3),
        configurar_random_forest(30, 8, max_features="all"),
    ]

    _, lda_tuning_tabla, _ = cross_validation_configuraciones_multiclase(
        desarrollo.reset_index(drop=True),
        folds,
        nombre_modelo="lda",
        grilla_configuraciones=grilla_configuracion_lda,
    )
    _, logistic_tuning_tabla, _ = cross_validation_configuraciones_multiclase(
        desarrollo.reset_index(drop=True),
        folds,
        nombre_modelo="multiclass_logistic",
        grilla_configuraciones=grilla_configuracion_logistica,
    )
    _, rf_tuning_tabla, _ = cross_validation_configuraciones_multiclase(
        desarrollo.reset_index(drop=True),
        folds,
        nombre_modelo="random_forest",
        grilla_configuraciones=grilla_configuracion_rf,
    )

    lda_tuning_tabla = lda_tuning_tabla.sort_values("covariance_ridge").reset_index(drop=True)
    logistic_tuning_tabla = logistic_tuning_tabla.sort_values("lambda_").reset_index(drop=True)
    mejor_config_lda = seleccionar_mejor_configuracion_multiclase(lda_tuning_tabla)
    mejor_config_logistica = seleccionar_mejor_configuracion_multiclase(logistic_tuning_tabla)
    mejor_config_rf = seleccionar_mejor_configuracion_multiclase(rf_tuning_tabla)

    return {
        "ajuste_lda_df": lda_tuning_tabla,
        "ajuste_logistica_multiclase_df": logistic_tuning_tabla,
        "ajuste_rf_df": rf_tuning_tabla,
        "mejor_config_lda": mejor_config_lda,
        "mejor_config_logistica_multiclase": mejor_config_logistica,
        "mejor_config_rf": mejor_config_rf,
    }

def evaluar_punto3(
    desarrollo: pd.DataFrame,
    folds,
    mejor_configuracion_lda: dict,
    mejor_configuracion_logistica: dict,
    mejor_configuracion_rf: dict,
) -> dict:
    evaluaciones = [
        evaluar_multiclase_out_of_fold(
            desarrollo.reset_index(drop=True),
            folds,
            nombre_modelo="lda",
            parametros_modelo={"covariance_ridge": float(mejor_configuracion_lda["covariance_ridge"])},
            etiqueta_modelo="LDA",
        ),
        evaluar_multiclase_out_of_fold(
            desarrollo.reset_index(drop=True),
            folds,
            nombre_modelo="multiclass_logistic",
            parametros_modelo={"lambda_": float(mejor_configuracion_logistica["lambda_"]), **PARAMETROS_LOGISTICA_BASE},
            etiqueta_modelo="Logistica multiclase",
        ),
        evaluar_multiclase_out_of_fold(
            desarrollo.reset_index(drop=True),
            folds,
            nombre_modelo="random_forest",
            parametros_modelo=parametros_rf_desde_config(mejor_configuracion_rf),
            etiqueta_modelo="Random Forest",
        ),
    ]

    resumen_modelos_tabla = pd.DataFrame([resultado["metrics"] for resultado in evaluaciones])
    metricas_por_clase_tabla = pd.concat([resultado["per_class_metrics"] for resultado in evaluaciones], ignore_index=True)
    importancia_features_rf_tabla = resumir_importancias_features(evaluaciones[2]["feature_importances"])

    curvas_modelos = [
        {
            "label": resultado["metrics"]["model"],
            "per_class_metrics": resultado["per_class_metrics"],
            "roc_curves": resultado["roc_curves"],
            "pr_curves": resultado["pr_curves"],
            "positive_rates": resultado["positive_rates"],
        }
        for resultado in evaluaciones
    ]

    class_rank = {"Insuficiente": 0, "Regular": 1, "Bueno": 2, "Excelente": 3}
    tabla_metricas_por_clase = (
        metricas_por_clase_tabla[["model", "class", "support", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]]
        .assign(class_rank=lambda df: df["class"].map(class_rank))
        .sort_values(["model", "class_rank"])
        .drop(columns="class_rank")
        .reset_index(drop=True)
        .rename(columns={"model": "modelo", "class": "clase", "support": "soporte"})
        .round(3)
    )
    tabla_resumen_modelos = (
        resumen_modelos_tabla.rename(columns={"model": "modelo", "overall_accuracy": "accuracy_global"})
        [["modelo", "accuracy_global", "macro_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_roc_auc", "macro_pr_auc"]]
        .round(3)
    )

    return {
        "evaluacion_lda": evaluaciones[0],
        "evaluacion_logistica_multiclase": evaluaciones[1],
        "evaluacion_rf": evaluaciones[2],
        "resumen_modelos_df": resumen_modelos_tabla,
        "metricas_por_clase_df": metricas_por_clase_tabla,
        "importancia_features_rf_df": importancia_features_rf_tabla,
        "curvas_modelos": curvas_modelos,
        "tabla_resumen_modelos_df": tabla_resumen_modelos,
        "tabla_metricas_por_clase_df": tabla_metricas_por_clase,
    }

def preparar_punto4(
    desarrollo: pd.DataFrame,
    rebalancing_methods_punto4=None,
    columna_objetivo: str = "rendimiento_binario_num",
    n_folds: int = 4,
) -> dict:
    rebalancing_methods_punto4 = list(rebalancing_methods_punto4 or REBALANCING_TECHNIQUES_ORDER.copy())
    folds = construir_indices_group_kfold(desarrollo["escuela"], n_folds=n_folds)
    point4_example_train_idx, _ = folds[0]
    ejemplo_balance = resumir_rebalancing_example(
        desarrollo.iloc[point4_example_train_idx].copy(),
        methods=rebalancing_methods_punto4,
        columna_objetivo=columna_objetivo,
    )
    return {
        "rebalancing_methods_punto4": rebalancing_methods_punto4,
        "folds_group_punto4": folds,
        "ejemplo_balance_punto4_df": ejemplo_balance,
    }

def armar_curvas_lambda_punto4(ajuste_punto4_tabla: pd.DataFrame, rebalancing_methods_punto4) -> list[dict]:
    return [
        {"label": REBALANCING_TECHNIQUE_LABELS[method], "df": ajuste_punto4_tabla[ajuste_punto4_tabla["technique"] == method].copy()}
        for method in rebalancing_methods_punto4
    ]

def ajustar_rebalancing_punto4(
    desarrollo: pd.DataFrame,
    folds,
    rebalancing_methods_punto4,
    columna_objetivo: str = "rendimiento_binario_num",
    grilla_lambda=None,
) -> dict:
    grilla_lambda = list(grilla_lambda or GRILLA_LAMBDA_BASE)

    ajuste_punto4_tabla = []
    mejores_filas = []
    for method in rebalancing_methods_punto4:
        _, resumen, best_lambda = evaluar_grilla_binaria(
            datos=desarrollo.reset_index(drop=True),
            folds=folds,
            grilla_lambda=grilla_lambda,
            columna_objetivo=columna_objetivo,
            parametros_modelo=None,
            fila_base={
                "technique": method,
                "technique_label": REBALANCING_TECHNIQUE_LABELS[method],
            },
            method=method,
        )
        ajuste_punto4_tabla.append(resumen)
        mejores_filas.append(
            {"technique": method, "technique_label": REBALANCING_TECHNIQUE_LABELS[method], "best_lambda": best_lambda}
        )
    ajuste_punto4_tabla = pd.concat(ajuste_punto4_tabla, ignore_index=True)
    mejor_lambda_punto4_tabla = pd.DataFrame(mejores_filas)

    ajuste_punto4_tabla = ajuste_punto4_tabla.sort_values(["technique_label", "lambda"]).reset_index(drop=True)
    return {
        "ajuste_punto4_df": ajuste_punto4_tabla,
        "mejor_lambda_punto4_df": mejor_lambda_punto4_tabla,
    }

def evaluar_punto4(
    desarrollo: pd.DataFrame,
    folds,
    point4_best_lambda_tabla: pd.DataFrame,
    columna_objetivo: str = "rendimiento_binario_num",
) -> dict:
    evaluaciones = [
        evaluar_rebalancing_out_of_fold(
            desarrollo.reset_index(drop=True),
            folds,
            method=fila["technique"],
            lambda_=float(fila["best_lambda"]),
            columna_objetivo=columna_objetivo,
        )
        for fila in point4_best_lambda_tabla.to_dict("records")
    ]

    metricas_punto4_tabla = pd.DataFrame([resultado["metrics"] for resultado in evaluaciones])
    tabla_metricas_punto4 = (
        metricas_punto4_tabla[["technique_label", "lambda", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]]
        .sort_values("technique_label")
        .reset_index(drop=True)
        .rename(columns={"technique_label": "method"})
        .round(3)
    )
    return {
        "metricas_punto4_df": metricas_punto4_tabla,
        "matrices_punto4": [
            {"label": resultado["metrics"]["technique_label"], "df": resultado["confusion_matrix"]}
            for resultado in evaluaciones
        ],
        "curvas_roc_punto4": [
            {"label": resultado["metrics"]["technique_label"], "df": resultado["roc_curve"], "auc": resultado["metrics"]["roc_auc"]}
            for resultado in evaluaciones
        ],
        "curvas_pr_punto4": [
            {
                "label": resultado["metrics"]["technique_label"],
                "df": resultado["pr_curve"],
                "auc": resultado["metrics"]["pr_auc"],
                "baseline": resultado["positive_rate"],
            }
            for resultado in evaluaciones
        ],
        "tabla_metricas_punto4": previsualizar_tabla(tabla_metricas_punto4, n=len(tabla_metricas_punto4), round_digits=3),
    }

def correr_punto5(
    desarrollo: pd.DataFrame,
    prueba: pd.DataFrame,
    eleccion_binaria: dict,
    eleccion_multiclase: dict,
    parametros_rf_referencia: dict,
    fracciones_curva_rf=None,
    rf_repeats: int = 4,
) -> dict:
    from configuracion import CARPETA_TABLAS_PUNTO_5

    fracciones_curva_rf = list(fracciones_curva_rf or FRACCIONES_CURVA_APRENDIZAJE_RF_BASE)
    tabla_seleccion = construir_tabla_seleccion_punto5(eleccion_binaria, eleccion_multiclase)
    vista_seleccion = tabla_seleccion.rename(
        columns={
            "task": "problema",
            "model": "modelo",
            "rebalancing": "rebalancing",
            "selection_strategy": "criterio_seleccion",
            "final_fit_data": "datos_fit_final",
            "selected_lambda": "lambda_elegido",
            "main_reason": "justificacion",
        }
    )

    resultado_binario = evaluar_pipeline_final_binario(
        datos_desarrollo=desarrollo,
        datos_prueba=prueba,
        lambda_=float(eleccion_binaria["lambda"]),
        method=eleccion_binaria["technique"],
        parametros_modelo=dict(eleccion_binaria.get("model_params") or PARAMETROS_LOGISTICA_BASE),
        columna_objetivo=eleccion_binaria.get("target_col", "rendimiento_binario_num"),
        etiqueta_estrategia=eleccion_binaria["validation_strategy"],
    )
    tabla_metricas_binarias = (
        resultado_binario["metrics_df"][["validation_strategy", "rebalancing_label", "lambda", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]]
        .rename(columns={"validation_strategy": "criterio_seleccion", "rebalancing_label": "rebalancing"})
        .round(3)
    )
    tabla_metricas_riesgo_binario = (
        resultado_binario["metrics_df"][["validation_strategy", "rebalancing_label", "lambda", "insuf_precision", "insuf_recall", "insuf_f1", "insuf_tp", "insuf_fp", "insuf_fn"]]
        .rename(
            columns={
                "validation_strategy": "criterio_seleccion",
                "rebalancing_label": "rebalancing",
                "insuf_precision": "precision_insuficiente",
                "insuf_recall": "recall_insuficiente",
                "insuf_f1": "f1_insuficiente",
                "insuf_tp": "tp_insuficiente",
                "insuf_fp": "fp_insuficiente",
                "insuf_fn": "fn_insuficiente",
            }
        )
        .round(3)
    )

    resultado_multiclase = evaluar_pipeline_final_multiclase(
        datos_desarrollo=desarrollo,
        datos_prueba=prueba,
        nombre_modelo=eleccion_multiclase["model_name"],
        parametros_modelo=dict(eleccion_multiclase["model_params"]),
        etiqueta_modelo=eleccion_multiclase["model_label"],
        validation_strategy=eleccion_multiclase["validation_strategy"],
        columna_objetivo=eleccion_multiclase.get("target_col", "rendimiento_ordinal"),
    )
    tabla_metricas_multiclase = (
        resultado_multiclase["metrics_df"][["validation_strategy", "overall_accuracy", "macro_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_roc_auc", "macro_pr_auc"]]
        .rename(columns={"validation_strategy": "criterio_seleccion", "overall_accuracy": "accuracy_global"})
        .round(3)
    )
    tabla_metricas_por_clase = (
        resultado_multiclase["per_class_metrics"][["class", "support", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]]
        .rename(columns={"class": "clase", "support": "soporte"})
        .round(3)
    )

    resultados_curva_rf = obtener_curva_aprendizaje_rf(
        datos_desarrollo=desarrollo,
        parametros_rf=dict(parametros_rf_referencia),
        fractions=fracciones_curva_rf,
        summary_path=CARPETA_TABLAS_PUNTO_5 / "rf_learning_curve_summary.csv",
        fitted_path=CARPETA_TABLAS_PUNTO_5 / "rf_learning_curve_fit.csv",
        estimate_path=CARPETA_TABLAS_PUNTO_5 / "rf_learning_curve_estimate.csv",
        repeats=rf_repeats,
        columna_objetivo=eleccion_multiclase.get("target_col", "rendimiento_ordinal"),
        columna_grupo="escuela",
        semilla=42,
    )

    return {
        "tabla_seleccion": previsualizar_tabla(vista_seleccion, n=len(vista_seleccion), round_digits=3),
        "tabla_metricas_binarias": previsualizar_tabla(tabla_metricas_binarias, n=1, round_digits=3),
        "tabla_metricas_riesgo_binario": previsualizar_tabla(tabla_metricas_riesgo_binario, n=1, round_digits=3),
        "matriz_confusion_binaria": resultado_binario["confusion_matrix"],
        "curva_roc_binaria": {
            "label": eleccion_binaria["model_label"],
            "df": resultado_binario["roc_curve"],
            "auc": float(resultado_binario["metrics"]["roc_auc"]),
        },
        "curva_pr_binaria": {
            "label": eleccion_binaria["model_label"],
            "df": resultado_binario["pr_curve"],
            "auc": float(resultado_binario["metrics"]["pr_auc"]),
            "baseline": float(resultado_binario["positive_rate"]),
        },
        "resultado_multiclase": resultado_multiclase,
        "tabla_metricas_multiclase": previsualizar_tabla(tabla_metricas_multiclase, n=1, round_digits=3),
        "tabla_metricas_por_clase_multiclase": previsualizar_tabla(
            tabla_metricas_por_clase,
            n=len(tabla_metricas_por_clase),
            round_digits=3,
        ),
        "curvas_multiclase": [
            {
                "label": eleccion_multiclase["model_label"],
                "per_class_metrics": resultado_multiclase["per_class_metrics"],
                "roc_curves": resultado_multiclase["roc_curves"],
                "pr_curves": resultado_multiclase["pr_curves"],
                "positive_rates": resultado_multiclase["positive_rates"],
            }
        ],
        "resultados_curva_rf": resultados_curva_rf,
        "tabla_estimacion_rf": previsualizar_tabla(
            resultados_curva_rf["estimate_df"].rename(
                columns={
                    "current_n": "n_actual",
                    "current_accuracy": "accuracy_actual",
                    "target_accuracy": "accuracy_objetivo",
                    "curve_intercept": "intercepto_curva",
                    "curve_slope": "pendiente_curva",
                    "fit_method": "metodo_ajuste",
                    "fit_points": "puntos_ajuste",
                    "fit_start_n": "n_inicial_ajuste",
                    "required_n": "n_requerido",
                    "additional_n": "n_adicional",
                    "finite_solution": "solucion_finita",
                }
            ),
            n=1,
            round_digits=3,
        ),
    }

def elegir_modelos_punto5(
    mejor_lambda_punto4_tabla: pd.DataFrame,
    mejor_config_logistica_multiclase: dict,
    mejor_config_rf: dict,
) -> dict:
    eleccion_binaria = {
        "model_label": "Logistica binaria L2",
        "rebalancing_label": "Cost re-weighting",
        "validation_strategy": "Temporal split como escenario final + prioridad a Insuficiente",
        "lambda": float(
            mejor_lambda_punto4_tabla.loc[
                mejor_lambda_punto4_tabla["technique"] == "cost_reweighting", "best_lambda"
            ].iloc[0]
        ),
        "technique": "cost_reweighting",
        "main_reason": (
            "En el caso binario me interesa no dejar pasar estudiantes en riesgo. "
            "Por eso elijo cost re-weighting: sube mucho la deteccion de Insuficientes "
            "sin duplicar filas ni generar datos sinteticos."
        ),
    }
    eleccion_multiclase = {
        "model_name": "multiclass_logistic",
        "model_label": "Logistica multiclase",
        "validation_strategy": "GroupKFold por escuela",
        "lambda": float(mejor_config_logistica_multiclase["lambda_"]),
        "model_params": {"lambda_": float(mejor_config_logistica_multiclase["lambda_"]), **PARAMETROS_LOGISTICA_BASE},
        "main_reason": (
            "En el punto 3 fue el modelo que mejor combino accuracy global y macro F1 "
            "cuando valide por escuela."
        ),
    }
    return {
        "eleccion_binaria": eleccion_binaria,
        "eleccion_multiclase": eleccion_multiclase,
        "parametros_rf": parametros_rf_desde_config(mejor_config_rf),
    }
