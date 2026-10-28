from __future__ import annotations

import numpy as np
import pandas as pd

from configuracion import (
    FEATURES_BINARIAS,
    FEATURES_NUMERICAS,
    ORDEN_ESCUELAS,
    ORDEN_OBJETIVO_BINARIO,
    ORDEN_OBJETIVO_MULTICLASE,
)


def resumir_estructura_dataset(datos: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dtype": datos.dtypes.astype(str),
            "n_missing": datos.isna().sum(),
            "pct_missing": datos.isna().mean().round(4),
            "n_unique": datos.nunique(dropna=False),
        }
    )


def resumir_problemas_limpieza(datos_crudos: pd.DataFrame) -> pd.DataFrame:
    lowercase_schools = (
        datos_crudos["escuela"].astype(str).str.strip()
        != datos_crudos["escuela"].astype(str).str.strip().str.upper()
    ).sum()
    invalid_previous_grade = (
        datos_crudos["nota_previa"].dropna().pipe(lambda s: (~s.between(0, 10)).sum())
    )

    return pd.DataFrame(
        [
            {
                "issue": "Etiquetas de escuela en minuscula",
                "count": int(lowercase_schools),
                "proposed_fix": "Unificar escuela a mayuscula.",
            },
            {
                "issue": "nota_previa fuera de rango [0, 10]",
                "count": int(invalid_previous_grade),
                "proposed_fix": "Auditar posible escala 0-100 y reescalar /10 si aplica.",
            },
        ]
    )


def resumir_escala_nota_previa(
    datos: pd.DataFrame,
    columna_valor: str = "nota_previa",
) -> pd.DataFrame:
    series = pd.to_numeric(datos[columna_valor], errors="coerce")
    valid = series[series.between(0, 10, inclusive="both")]
    suspect = series[(series > 10) & (series <= 100)]
    suspect_scaled = suspect / 10

    grupos = {
        "valid_0_10": valid,
        "suspect_10_100": suspect,
        "suspect_div_10": suspect_scaled,
    }

    filas = []
    for nombre_grupo, serie_grupo in grupos.items():
        filas.append(
            {
                "group": nombre_grupo,
                "n": int(serie_grupo.shape[0]),
                "mean": serie_grupo.mean(),
                "median": serie_grupo.median(),
                "std": serie_grupo.std(),
                "min": serie_grupo.min(),
                "q1": serie_grupo.quantile(0.25),
                "q3": serie_grupo.quantile(0.75),
                "max": serie_grupo.max(),
            }
        )

    return pd.DataFrame(filas).round(4)


def comparar_metricas_escala_nota_previa(
    datos: pd.DataFrame,
    columna_valor: str = "nota_previa",
) -> pd.DataFrame:
    resumen = (
        resumir_escala_nota_previa(datos, columna_valor=columna_valor)
        .set_index("group")
    )
    orden_metricas = ["mean", "median", "std", "q1", "q3"]
    valid_metrics = resumen.loc["valid_0_10", orden_metricas]
    suspect_metrics = resumen.loc["suspect_10_100", orden_metricas]
    scaled_metrics = resumen.loc["suspect_div_10", orden_metricas]

    comparison_tabla = pd.DataFrame(
        {
            "metric": orden_metricas,
            "valid_0_10": valid_metrics.values,
            "suspect_10_100": suspect_metrics.values,
            "suspect_div_10": scaled_metrics.values,
            "raw_ratio_vs_valid": (suspect_metrics / valid_metrics).values,
            "scaled_diff_vs_valid": (scaled_metrics - valid_metrics).values,
        }
    )
    return comparison_tabla.round(4)


def resumir_impacto_reescalado_nota_previa(
    datos: pd.DataFrame,
    columna_valor: str = "nota_previa",
) -> pd.DataFrame:
    series = pd.to_numeric(datos[columna_valor], errors="coerce")
    n_missing_raw = int(series.isna().sum())
    n_valid_0_10 = int(series.between(0, 10, inclusive="both").sum())
    n_suspect_10_100 = int(((series > 10) & (series <= 100)).sum())

    non_missing = n_valid_0_10 + n_suspect_10_100
    share_non_missing = n_suspect_10_100 / non_missing if non_missing else np.nan

    return pd.DataFrame(
        [
            {
                "n_total": int(series.shape[0]),
                "n_missing_raw": n_missing_raw,
                "n_valid_0_10": n_valid_0_10,
                "n_suspect_10_100": n_suspect_10_100,
                "share_suspect_non_missing": share_non_missing,
                "share_suspect_total": n_suspect_10_100 / series.shape[0],
                "missing_if_convert_to_nan": n_missing_raw + n_suspect_10_100,
                "missing_if_rescale": n_missing_raw,
            }
        ]
    ).round(4)


def resumir_valores_faltantes(datos: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "feature": datos.columns,
            "n_missing": datos.isna().sum().values,
            "pct_missing": datos.isna().mean().values,
        }
    )
    return summary.sort_values(["pct_missing", "feature"], ascending=[False, True])


def resumir_features_numericas(datos: pd.DataFrame, columnas_numericas=None) -> pd.DataFrame:
    columnas_numericas = columnas_numericas or FEATURES_NUMERICAS
    return (
        datos[columnas_numericas]
        .describe()
        .T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        .round(3)
    )


def resumir_outliers_iqr(datos: pd.DataFrame, columnas_numericas=None) -> pd.DataFrame:
    columnas_numericas = columnas_numericas or FEATURES_NUMERICAS
    filas = []

    for col in columnas_numericas:
        series = datos[col].dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()

        filas.append(
            {
                "feature": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "iqr_lower": lower,
                "iqr_upper": upper,
                "n_outliers_iqr": int(outliers),
            }
        )

    return pd.DataFrame(filas).round(3).sort_values(
        "n_outliers_iqr",
        ascending=False,
    )


def resumir_chequeos_dominio(datos: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "nota_previa_gt_10": (datos["nota_previa"] > 10).sum(),
        "nota_previa_lt_0": (datos["nota_previa"] < 0).sum(),
        "asistencia_gt_100": (datos["asistencia"] > 100).sum(),
        "asistencia_lt_0": (datos["asistencia"] < 0).sum(),
        "nivel_socioeconomico_out_of_range": (
            (datos["nivel_socioeconomico"] < 0) | (datos["nivel_socioeconomico"] > 1)
        ).sum(),
        "acceso_internet_not_binary": (~datos["acceso_internet"].isin([0, 1])).sum(),
    }

    return (
        pd.Series(checks, name="count")
        .rename_axis("check")
        .reset_index()
        .sort_values("count", ascending=False)
    )


def distribucion_clases(series: pd.Series, orden_clases) -> pd.DataFrame:
    conteos = series.value_counts(dropna=False).reindex(orden_clases, fill_value=0)
    proporciones = conteos / conteos.sum()

    return pd.DataFrame(
        {
            "count": conteos,
            "proportion": proporciones.round(4),
        }
    )


def resumir_distribuciones_objetivo(datos: pd.DataFrame):
    return {
        "multiclass": distribucion_clases(datos["rendimiento"], ORDEN_OBJETIVO_MULTICLASE),
        "binary": distribucion_clases(datos["rendimiento_binario"], ORDEN_OBJETIVO_BINARIO),
    }


def distribucion_objetivo_agrupada(
    datos: pd.DataFrame,
    columna_grupo: str,
    columna_objetivo: str,
    normalize: bool = True,
) -> pd.DataFrame:
    normalize_mode = "index" if normalize else False
    table = pd.crosstab(
        datos[columna_grupo],
        datos[columna_objetivo],
        normalize=normalize_mode,
    )
    return table.round(4)


def resumir_features_por_escuela(datos: pd.DataFrame) -> pd.DataFrame:
    return (
        datos.groupby("escuela", observed=True)[FEATURES_NUMERICAS + FEATURES_BINARIAS]
        .mean()
        .round(3)
        .reindex(ORDEN_ESCUELAS)
    )

def correlaciones_feature_target_por_escuela(
    datos: pd.DataFrame,
    columna_objetivo: str = "rendimiento_ordinal",
    columnas_numericas=None,
) -> pd.DataFrame:
    columnas_numericas = columnas_numericas or (FEATURES_NUMERICAS + FEATURES_BINARIAS)
    correlations = {}

    for escuela, datos_grupo in datos.groupby("escuela", observed=True):
        clave_escuela = str(escuela)
        corr = (
            datos_grupo[columnas_numericas + [columna_objetivo]]
            .corr(numeric_only=True)[columna_objetivo]
            .drop(columna_objetivo)
        )
        correlations[clave_escuela] = corr

    return pd.DataFrame(correlations).round(3)


def resumir_rango_correlaciones(correlaciones: pd.DataFrame) -> pd.DataFrame:
    ranges = (correlaciones.max(axis=1) - correlaciones.min(axis=1)).sort_values(ascending=False)
    return ranges.rename("range").reset_index().rename(columns={"index": "feature"})


def tasa_aprobacion_por_escuela_semestre(datos: pd.DataFrame) -> pd.DataFrame:
    return (
        datos.pivot_table(
            index="escuela",
            columns="semestre",
            values="rendimiento_binario_num",
            aggfunc="mean",
            observed=True,
        )
        .round(3)
        .reindex(index=ORDEN_ESCUELAS)
    )

