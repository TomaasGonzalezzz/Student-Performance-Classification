from __future__ import annotations

import pandas as pd

from configuracion import FEATURES_BINARIAS, FEATURES_CATEGORICAS, FEATURES_NUMERICAS


def ajustar_imputador_mediana_por_grupo(
    datos: pd.DataFrame,
    columnas_numericas=None,
    columna_grupo: str = "escuela",
):
    columnas_numericas = columnas_numericas or (FEATURES_NUMERICAS + FEATURES_BINARIAS)
    # Guardamos tanto la mediana por escuela como la global para poder cubrir
    # el caso en que aparezca un grupo nuevo o una escuela sin informacion.
    group_medians = datos.groupby(columna_grupo, observed=True)[columnas_numericas].median()
    global_medians = datos[columnas_numericas].median()

    return {
        "group_col": columna_grupo,
        "numeric_cols": columnas_numericas,
        "group_medians": group_medians,
        "global_medians": global_medians,
    }


def aplicar_imputador_mediana_por_grupo(datos: pd.DataFrame, estado_imputador) -> pd.DataFrame:
    transformed = datos.copy()
    columna_grupo = estado_imputador["group_col"]
    columnas_numericas = estado_imputador["numeric_cols"]
    group_medians = estado_imputador["group_medians"]
    global_medians = estado_imputador["global_medians"]

    group_values = transformed[columna_grupo].astype(str)

    for col in columnas_numericas:
        # Primero usamos la referencia especifica de la escuela.
        transformed[col] = transformed[col].fillna(group_values.map(group_medians[col]))
        # Si todavia falta algo, caemos a la mediana global aprendida en train.
        transformed[col] = transformed[col].fillna(global_medians[col])

    return transformed


def ajustar_estandarizador(datos: pd.DataFrame, columnas_numericas=None):
    columnas_numericas = columnas_numericas or FEATURES_NUMERICAS
    means = datos[columnas_numericas].mean()
    # Reemplazamos desvio 0 por 1 para no dividir por cero en columnas constantes.
    stds = datos[columnas_numericas].std(ddof=0).replace(0, 1.0)

    return {
        "numeric_cols": columnas_numericas,
        "means": means,
        "stds": stds,
    }


def aplicar_estandarizador(datos: pd.DataFrame, estado_estandarizador) -> pd.DataFrame:
    transformed = datos.copy()
    columnas_numericas = estado_estandarizador["numeric_cols"]

    transformed[columnas_numericas] = (
        transformed[columnas_numericas] - estado_estandarizador["means"]
    ) / estado_estandarizador["stds"]

    return transformed


def construir_matriz_features_modelo(
    datos: pd.DataFrame,
    incluir_escuela: bool = True,
    incluir_semestre: bool = True,
    omitir_primera: bool = True,
) -> pd.DataFrame:
    partes_features = [
        datos[FEATURES_NUMERICAS].astype(float),
        datos[FEATURES_BINARIAS].astype(float),
    ]

    columnas_categoricas = []
    if incluir_escuela:
        columnas_categoricas.append("escuela")
    if incluir_semestre:
        columnas_categoricas.append("semestre")

    if columnas_categoricas:
        # Las categoricas se incorporan como dummies para que todos los modelos
        # trabajen con una matriz numerica.
        dummies = pd.get_dummies(
            datos[columnas_categoricas],
            drop_first=omitir_primera,
            dtype=float,
        )
        partes_features.append(dummies)

    return pd.concat(partes_features, axis=1)


def ajustar_preprocesador_modelo(
    datos: pd.DataFrame,
    incluir_escuela: bool = True,
    incluir_semestre: bool = True,
    omitir_primera: bool = True,
):
    # Este orden replica exactamente el pipeline que despues se aplica en cada fold.
    estado_imputador = ajustar_imputador_mediana_por_grupo(datos)
    imputed_tabla = aplicar_imputador_mediana_por_grupo(datos, estado_imputador)
    estado_estandarizador = ajustar_estandarizador(imputed_tabla, columnas_numericas=FEATURES_NUMERICAS)
    scaled_tabla = aplicar_estandarizador(imputed_tabla, estado_estandarizador)
    feature_tabla = construir_matriz_features_modelo(
        scaled_tabla,
        incluir_escuela=incluir_escuela,
        incluir_semestre=incluir_semestre,
        omitir_primera=omitir_primera,
    )

    return {
        "imputer_state": estado_imputador,
        "scaler_state": estado_estandarizador,
        "include_school": incluir_escuela,
        "include_semester": incluir_semestre,
        "drop_first": omitir_primera,
        "feature_columns": feature_tabla.columns.tolist(),
    }


def aplicar_preprocesador_modelo(datos: pd.DataFrame, estado_preprocesador) -> pd.DataFrame:
    transformed_tabla = aplicar_imputador_mediana_por_grupo(datos, estado_preprocesador["imputer_state"])
    transformed_tabla = aplicar_estandarizador(
        transformed_tabla,
        estado_preprocesador["scaler_state"],
    )
    feature_tabla = construir_matriz_features_modelo(
        transformed_tabla,
        incluir_escuela=estado_preprocesador["include_school"],
        incluir_semestre=estado_preprocesador["include_semester"],
        omitir_primera=estado_preprocesador["drop_first"],
    )
    # Reindexamos para que train, validation y test queden con exactamente las
    # mismas columnas, incluso si en alguno falta una dummy.
    return feature_tabla.reindex(
        columns=estado_preprocesador["feature_columns"],
        fill_value=0.0,
    )
