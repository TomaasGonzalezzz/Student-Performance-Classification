from __future__ import annotations

import numpy as np
import pandas as pd

from configuracion import (
    FEATURES_MODELO,
    FEATURES_BINARIAS,
    CARPETA_DATOS_CRUDOS,
    ORDEN_ESCUELAS,
    ORDEN_SEMESTRES,
    ORDEN_OBJETIVO_BINARIO,
    ORDEN_OBJETIVO_MULTICLASE,
    MAPA_OBJETIVO_ORDINAL,
)


def cargar_dataset_crudo(split: str) -> pd.DataFrame:
    ruta = CARPETA_DATOS_CRUDOS / f"rendimiento_estudiantes_{split}.csv"
    return pd.read_csv(ruta)


def limpiar_dataframe_estudiantes(datos: pd.DataFrame) -> pd.DataFrame:
    datos_limpios = datos.copy()

    for col in FEATURES_MODELO:
        if col in datos_limpios.columns and col not in {"escuela", "semestre"}:
            datos_limpios[col] = pd.to_numeric(datos_limpios[col], errors="coerce")

    datos_limpios["escuela"] = datos_limpios["escuela"].astype(str).str.strip().str.upper()
    datos_limpios["semestre"] = datos_limpios["semestre"].astype(str).str.strip()
    datos_limpios["rendimiento"] = datos_limpios["rendimiento"].astype(str).str.strip()

    scaled_previous_grade = (
        datos_limpios["nota_previa"].notna()
        & (datos_limpios["nota_previa"] > 10)
        & (datos_limpios["nota_previa"] <= 100)
    )
    # Si quedo cargada en escala 0-100, la llevamos a 0-10.
    datos_limpios.loc[scaled_previous_grade, "nota_previa"] = (
        datos_limpios.loc[scaled_previous_grade, "nota_previa"] / 10
    )

    invalid_previous_grade = (
        datos_limpios["nota_previa"].notna()
        & ~datos_limpios["nota_previa"].between(0, 10)
    )
    datos_limpios.loc[invalid_previous_grade, "nota_previa"] = np.nan

    datos_limpios["escuela"] = pd.Categorical(
        datos_limpios["escuela"],
        categories=ORDEN_ESCUELAS,
        ordered=True,
    )
    datos_limpios["semestre"] = pd.Categorical(
        datos_limpios["semestre"],
        categories=ORDEN_SEMESTRES,
        ordered=True,
    )
    datos_limpios["rendimiento"] = pd.Categorical(
        datos_limpios["rendimiento"],
        categories=ORDEN_OBJETIVO_MULTICLASE,
        ordered=True,
    )

    datos_limpios["rendimiento_binario"] = np.where(
        datos_limpios["rendimiento"].astype(str) == "Insuficiente",
        "Insuficiente",
        "Aprobado",
    )
    datos_limpios["rendimiento_binario"] = pd.Categorical(
        datos_limpios["rendimiento_binario"],
        categories=ORDEN_OBJETIVO_BINARIO,
        ordered=True,
    )
    datos_limpios["rendimiento_binario_num"] = (
        datos_limpios["rendimiento_binario"].astype(str) == "Aprobado"
    ).astype(int)
    datos_limpios["rendimiento_ordinal"] = (
        datos_limpios["rendimiento"].astype(str).map(MAPA_OBJETIVO_ORDINAL).astype(int)
    )

    for col in FEATURES_BINARIAS:
        datos_limpios[col] = datos_limpios[col].astype("Int64")

    return datos_limpios

