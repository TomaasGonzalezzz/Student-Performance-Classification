from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configuracion import (
    EXPORTAR_FIGURAS,
    FEATURES_NUMERICAS,
    ORDEN_ESCUELAS,
    ORDEN_OBJETIVO_BINARIO,
    ORDEN_OBJETIVO_MULTICLASE,
)
from analisis_exploratorio import comparar_metricas_escala_nota_previa, distribucion_objetivo_agrupada
from utilidades import preparar_ruta_salida

def configurar_estilo_graficos():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 12

def guardar_figura(fig, ruta_guardado=None):
    if ruta_guardado is not None and EXPORTAR_FIGURAS:
        ruta_guardado = preparar_ruta_salida(ruta_guardado)
        fig.savefig(ruta_guardado, dpi=160, bbox_inches="tight")
    return fig

def previsualizar_tabla(
    datos: pd.DataFrame,
    n: int = 6,
    columnas=None,
    round_digits: int | None = 3,
) -> pd.DataFrame:
    """Devuelve una vista corta de una tabla para mostrar en el notebook."""
    preview_tabla = datos.copy()
    if columnas is not None:
        preview_tabla = preview_tabla.loc[:, columnas]
    preview_tabla = preview_tabla.head(n)

    if round_digits is not None:
        columnas_numericas = preview_tabla.select_dtypes(include="number").columns
        preview_tabla.loc[:, columnas_numericas] = preview_tabla.loc[:, columnas_numericas].round(round_digits)

    return preview_tabla

def formatear_marca_regularizacion(valor: float) -> str:
    """Formatea valores en escala logaritmica de manera compacta."""
    valor = float(valor)
    if valor == 0:
        return "0"
    if valor >= 1:
        if abs(valor - round(valor)) < 1e-12:
            return str(int(round(valor)))
        return f"{valor:.2f}".rstrip("0").rstrip(".")
    return f"{valor:.0e}"

def seleccionar_valores_marca_log(valores) -> list[float]:
    """Elige marcas espaciadas por decadas para un eje logaritmico positivo."""
    value_array = np.asarray(valores, dtype=float)
    value_array = value_array[value_array > 0]
    if value_array.size == 0:
        return []

    min_exponent = int(np.floor(np.log10(value_array.min())))
    max_exponent = int(np.ceil(np.log10(value_array.max())))
    ticks = []

    for exponent in range(min_exponent, max_exponent + 1):
        tick_value = float(10 ** exponent)
        if value_array.min() <= tick_value <= value_array.max():
            ticks.append(tick_value)

    min_value = float(value_array.min())
    max_value = float(value_array.max())
    if not ticks or not np.isclose(ticks[0], min_value):
        ticks.insert(0, min_value)
    if not np.isclose(ticks[-1], max_value):
        ticks.append(max_value)

    deduped_ticks = []
    for tick_value in ticks:
        if not deduped_ticks or not np.isclose(deduped_ticks[-1], tick_value):
            deduped_ticks.append(tick_value)

    return deduped_ticks

def aplicar_marcas_log_espaciadas(ax, valores):
    """Aplica marcas legibles sobre un eje x en escala logaritmica."""
    tick_values = seleccionar_valores_marca_log(valores)
    ax.set_xscale("log")
    ax.set_xticks(tick_values)
    ax.set_xticklabels(
        [formatear_marca_regularizacion(valor) for valor in tick_values],
        rotation=35,
        ha="right",
    )
    ax.tick_params(axis="x", labelsize=9)

def anotar_valores_barras(ax, orientation: str = "horizontal", decimals: int = 3):
    """Anota el valor numerico al final de cada barra."""
    if orientation == "horizontal":
        x_limit = ax.get_xlim()[1]
        offset = x_limit * 0.012
        for patch in ax.patches:
            width = patch.get_width()
            y_center = patch.get_y() + patch.get_height() / 2
            ax.text(
                width + offset,
                y_center,
                f"{width:.{decimals}f}",
                va="center",
                ha="left",
                fontsize=9,
            )
    else:
        y_limit = ax.get_ylim()[1]
        offset = y_limit * 0.015
        for patch in ax.patches:
            height = patch.get_height()
            x_center = patch.get_x() + patch.get_width() / 2
            ax.text(
                x_center,
                height + offset,
                f"{height:.{decimals}f}",
                va="bottom",
                ha="center",
                fontsize=9,
            )

def dibujar_matriz_confusion(ax, matriz_confusion: pd.DataFrame, titulo: str):
    sns.heatmap(matriz_confusion.astype(int), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Valor real")

def dibujar_curva(ax, spec: dict, x_col: str, y_col: str, color, etiqueta: str | None = None):
    etiqueta = etiqueta or f"{spec['label']} (AUC={spec['auc']:.3f})"
    ax.plot(spec["df"][x_col], spec["df"][y_col], linewidth=2, color=color, label=etiqueta)

def cerrar_eje_curva(ax, titulo: str, xlabel: str, ylabel: str, diagonal=False, baseline=None, legend_loc="best"):
    if diagonal:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", color="gray", linewidth=1.0, label=f"baseline = {baseline:.3f}")
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend_loc is not None:
        ax.legend(loc=legend_loc)

def valor_metrica_clase(spec: dict, etiqueta_clase: str, metrica: str) -> float:
    tabla = spec["per_class_metrics"]
    return float(tabla.loc[tabla["class"] == etiqueta_clase, metrica].iloc[0])

def anotar_resumen_curvas(ax, titulo: str, lineas):
    ax.text(
        0.98,
        0.03,
        titulo + "\n" + "\n".join(lineas),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#b0b0b0", alpha=0.9),
    )

def etiqueta_metrica_multiclase(metrica: str) -> str:
    """Etiqueta legible para los nombres de metricas multiclase."""
    etiquetas_metricas = {
        "overall_accuracy": "Accuracy global",
        "macro_accuracy": "Accuracy macro",
        "accuracy": "Accuracy",
        "macro_precision": "Precision macro",
        "precision": "Precision",
        "macro_recall": "Recall macro",
        "recall": "Recall",
        "macro_f1": "Macro F1",
        "f1_score": "F1",
        "macro_roc_auc": "Macro ROC-AUC",
        "roc_auc": "ROC-AUC",
        "macro_pr_auc": "Macro PR-AUC",
        "pr_auc": "PR-AUC",
    }
    return etiquetas_metricas.get(metrica, metrica)

def graficar_panel_matrices_confusion(confusion_specs, titulo, layout=(1, 3), figsize=(15, 4.5), ruta_guardado=None):
    fig, axes = plt.subplots(*layout, figsize=figsize)
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, spec in zip(axes, confusion_specs):
        dibujar_matriz_confusion(ax, spec["df"], spec["label"])
    for ax in axes[len(confusion_specs):]:
        ax.axis("off")
    fig.suptitle(titulo, y=1.02 if len(axes) > 1 else 1.0)
    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_panel_curvas_binarias(curve_specs, x_col, y_col, titulo, xlabel, ylabel, baseline_key=None, legend_loc="lower right", ruta_guardado=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.reshape(-1)
    palette = ["#2f6c8f", "#bc4749", "#4c956c"]
    for ax, color, spec in zip(axes[:3], palette, curve_specs):
        dibujar_curva(ax, spec, x_col, y_col, color, etiqueta=f"AUC = {spec['auc']:.3f}")
        cerrar_eje_curva(ax, spec["label"], xlabel, ylabel, diagonal=(x_col == "fpr"), baseline=spec.get(baseline_key) if baseline_key else None, legend_loc=legend_loc)
    combinado = axes[3]
    for color, spec in zip(palette, curve_specs):
        dibujar_curva(combinado, spec, x_col, y_col, color)
    cerrar_eje_curva(combinado, "Comparacion conjunta", xlabel, ylabel, diagonal=(x_col == "fpr"), legend_loc=legend_loc)
    fig.suptitle(titulo, y=1.02)
    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_curvas_superpuestas(curve_specs, x_col, y_col, titulo, xlabel, ylabel, baseline=None, legend_title="Serie", ruta_guardado=None):
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    for color, spec in zip(sns.color_palette("tab10", n_colors=len(curve_specs)), curve_specs):
        dibujar_curva(ax, spec, x_col, y_col, color)
    cerrar_eje_curva(ax, titulo, xlabel, ylabel, diagonal=(x_col == "fpr"), baseline=baseline)
    ax.legend(title=legend_title, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return guardar_figura(fig, ruta_guardado)

def graficar_curva_ajuste_regularizacion(
    ajuste: pd.DataFrame,
    x_col: str,
    y_col: str = "mean_macro_f1",
    error_col: str | None = "std_macro_f1",
    title: str = "Macro Validation F1 vs regularizacion",
    x_label: str = "regularizacion",
    color: str = "#2f6c8f",
    ruta_guardado=None,
):
    """Grafica una curva de ajuste de regularizacion en eje x logaritmico."""
    datos_grafico = ajuste.sort_values(x_col).copy()
    x_values = datos_grafico[x_col].astype(float).to_numpy()
    y_values = datos_grafico[y_col].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(
        x_values,
        y_values,
        marker="o",
        linewidth=2,
        color=color,
    )

    if error_col is not None and error_col in datos_grafico.columns:
        y_error = datos_grafico[error_col].fillna(0.0).astype(float).to_numpy()
        ax.fill_between(
            x_values,
            y_values - y_error,
            y_values + y_error,
            color=color,
            alpha=0.12,
        )

    best_idx = int(np.argmax(y_values))
    best_x = float(x_values[best_idx])
    best_y = float(y_values[best_idx])
    ax.scatter([best_x], [best_y], color="#bc4749", s=60, zorder=3)
    ax.axvline(best_x, color="#bc4749", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.annotate(
        f"mejor = {best_x:g}\nMacro F1 = {best_y:.3f}",
        xy=(best_x, best_y),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bc4749", alpha=0.9),
    )

    aplicar_marcas_log_espaciadas(ax, x_values)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Macro F1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def paleta_escuelas():
    colors = sns.color_palette("tab10", n_colors=len(ORDEN_ESCUELAS))
    return dict(zip(ORDEN_ESCUELAS, colors))

def graficar_valores_faltantes(faltantes: pd.DataFrame, ruta_guardado=None):
    datos_grafico = faltantes[faltantes["n_missing"] > 0].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=datos_grafico,
        x="feature",
        y="pct_missing",
        color="#2f6c8f",
        ax=ax,
    )
    ax.set_title("Porcentaje de valores faltantes")
    ax.set_xlabel("")
    ax.set_ylabel("Proporcion")
    ax.set_ylim(0, datos_grafico["pct_missing"].max() * 1.15)
    ax.tick_params(axis="x", rotation=45)

    for patch, valor in zip(ax.patches, datos_grafico["pct_missing"]):
        ax.annotate(
            f"{valor * 100:.1f}%",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_distribuciones_numericas(datos: pd.DataFrame, columnas_numericas=None, ruta_guardado=None):
    columnas_numericas = columnas_numericas or FEATURES_NUMERICAS
    n_cols = 3
    n_rows = int(np.ceil(len(columnas_numericas) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, columnas_numericas):
        sns.histplot(datos[col], bins=25, kde=True, color="#4c956c", ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")

    for ax in axes[len(columnas_numericas):]:
        ax.axis("off")

    fig.suptitle("Distribucion de features numericas", y=1.02)
    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_auditoria_escala_nota_previa(
    datos_crudos: pd.DataFrame,
    columna_valor: str = "nota_previa",
    ruta_guardado=None,
):
    series = pd.to_numeric(datos_crudos[columna_valor], errors="coerce")
    valid = series[series.between(0, 10, inclusive="both")]
    suspect_scaled = series[(series > 10) & (series <= 100)] / 10
    metric_tabla = comparar_metricas_escala_nota_previa(datos_crudos, columna_valor=columna_valor)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    bins = np.linspace(0, 10, 21)

    sns.histplot(
        valid,
        bins=bins,
        stat="density",
        kde=True,
        color="#4c956c",
        alpha=0.4,
        label="notas validas 0-10",
        ax=axes[0],
    )
    sns.histplot(
        suspect_scaled,
        bins=bins,
        stat="density",
        kde=True,
        color="#bc4749",
        alpha=0.4,
        label="notas >10 reescaladas /10",
        ax=axes[0],
    )
    axes[0].set_title("nota_previa: validas vs sospechosas reescaladas")
    axes[0].set_xlabel("nota_previa")
    axes[0].set_ylabel("Densidad")
    axes[0].legend()

    sns.barplot(
        data=metric_tabla,
        x="metric",
        y="raw_ratio_vs_valid",
        color="#8da9c4",
        ax=axes[1],
    )
    axes[1].axhline(10, color="#bc4749", linestyle="--", linewidth=1.8)
    axes[1].set_title("Ratios crudos frente al grupo valido")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Ratio")
    axes[1].set_ylim(0, max(11.5, metric_tabla["raw_ratio_vs_valid"].max() * 1.1))

    for patch, valor in zip(axes[1].patches, metric_tabla["raw_ratio_vs_valid"]):
        axes[1].annotate(
            f"{valor:.2f}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_medianas_imputacion_por_escuela(
    datos: pd.DataFrame,
    features_con_faltantes=None,
    ruta_guardado=None,
):
    features_con_faltantes = features_con_faltantes or [
        "horas_estudio",
        "nota_previa",
        "horas_sueno",
        "participacion",
        "nivel_socioeconomico",
    ]
    palette = paleta_escuelas()
    school_medians = (
        datos.groupby("escuela", observed=True)[features_con_faltantes]
        .median()
        .reindex(ORDEN_ESCUELAS)
    )
    global_medians = datos[features_con_faltantes].median()

    n_cols = 2
    n_rows = int(np.ceil(len(features_con_faltantes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.8 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, features_con_faltantes):
        valores = school_medians[col]
        sns.barplot(
            x=valores.index,
            y=valores.values,
            hue=valores.index,
            palette=[palette[label] for label in valores.index],
            legend=False,
            ax=ax,
        )
        ax.axhline(
            global_medians[col],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"mediana global = {global_medians[col]:.2f}",
        )
        ax.set_title(f"Medianas por escuela - {col}")
        ax.set_xlabel("")
        ax.set_ylabel("Mediana")
        ax.legend(loc="best")

    for ax in axes[len(features_con_faltantes):]:
        ax.axis("off")

    fig.suptitle(
        "Referencia para la imputacion: medianas por escuela vs mediana global",
        y=1.02,
    )
    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_distribuciones_objetivo(datos: pd.DataFrame, ruta_guardado=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.countplot(
        data=datos,
        x="rendimiento",
        hue="rendimiento",
        order=ORDEN_OBJETIVO_MULTICLASE,
        hue_order=ORDEN_OBJETIVO_MULTICLASE,
        palette="crest",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Target multiclase")
    axes[0].set_xlabel("")

    sns.countplot(
        data=datos,
        x="rendimiento_binario",
        hue="rendimiento_binario",
        order=ORDEN_OBJETIVO_BINARIO,
        hue_order=ORDEN_OBJETIVO_BINARIO,
        palette="flare",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Target binario")
    axes[1].set_xlabel("")

    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_boxplots_numericos_por_escuela(datos: pd.DataFrame, columnas_numericas=None, ruta_guardado=None):
    columnas_numericas = columnas_numericas or FEATURES_NUMERICAS
    n_cols = 3
    n_rows = int(np.ceil(len(columnas_numericas) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, columnas_numericas):
        sns.boxplot(
            data=datos,
            x="escuela",
            y=col,
            order=ORDEN_ESCUELAS,
            color="#8da9c4",
            showfliers=False,
            ax=ax,
        )
        ax.set_title(col)
        ax.set_xlabel("")

    for ax in axes[len(columnas_numericas):]:
        ax.axis("off")

    fig.suptitle("Distribuciones de features numericas por escuela", y=1.02)
    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_distribucion_objetivo_por_escuela(datos: pd.DataFrame, ruta_guardado=None):
    school_multiclass = distribucion_objetivo_agrupada(
        datos,
        columna_grupo="escuela",
        columna_objetivo="rendimiento",
        normalize=True,
    ).reindex(ORDEN_ESCUELAS)

    school_binary = distribucion_objetivo_agrupada(
        datos,
        columna_grupo="escuela",
        columna_objetivo="rendimiento_binario",
        normalize=True,
    ).reindex(ORDEN_ESCUELAS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    school_multiclass.plot(
        kind="bar",
        stacked=True,
        colormap="viridis",
        ax=axes[0],
    )
    axes[0].set_title("Distribucion del target por escuela")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Proporcion")
    axes[0].legend(title="rendimiento", bbox_to_anchor=(1.0, 1.0))

    school_binary.plot(
        kind="bar",
        stacked=True,
        color=["#bc4749", "#588157"],
        ax=axes[1],
    )
    axes[1].set_title("Version binaria por escuela")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Proporcion")
    axes[1].legend(title="target binario", bbox_to_anchor=(1.0, 1.0))

    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_distribucion_objetivo_por_semestre(datos: pd.DataFrame, ruta_guardado=None):
    multiclass = distribucion_objetivo_agrupada(
        datos,
        columna_grupo="semestre",
        columna_objetivo="rendimiento",
        normalize=True,
    )
    binary = distribucion_objetivo_agrupada(
        datos,
        columna_grupo="semestre",
        columna_objetivo="rendimiento_binario",
        normalize=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    multiclass.plot(kind="bar", stacked=True, colormap="mako", ax=axes[0])
    axes[0].set_title("Distribucion del target por semestre")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Proporcion")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(title="rendimiento", bbox_to_anchor=(1.0, 1.0))

    binary.plot(
        kind="bar",
        stacked=True,
        color=["#bc4749", "#588157"],
        ax=axes[1],
    )
    axes[1].set_title("Version binaria por semestre")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Proporcion")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(title="target binario", bbox_to_anchor=(1.0, 1.0))

    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_heatmap_tasa_aprobacion(aprobacion: pd.DataFrame, ruta_guardado=None):
    datos_grafico = aprobacion.astype(float)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        datos_grafico,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Tasa de aprobacion por escuela y semestre")
    ax.set_xlabel("Semestre")
    ax.set_ylabel("Escuela")
    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_heatmap_correlacion_feature_objetivo(correlaciones: pd.DataFrame, ruta_guardado=None):
    datos_grafico = correlaciones.astype(float)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        datos_grafico,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlacion feature-target ordinal por escuela")
    ax.set_xlabel("Escuela")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_panel_ajuste_lambda(
    tuning_specs,
    ruta_guardado=None,
):
    """Muestra tres curvas de lambda separadas y una comparacion conjunta."""
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2))
    axes = axes.reshape(-1)
    palette = ["#2f6c8f", "#bc4749", "#4c956c"]
    legend_handles = []
    legend_labels = []

    for ax, color, spec in zip(axes[:3], palette, tuning_specs):
        ajuste = spec["df"]
        label = spec["label"]
        use_mean_columns = spec.get("use_mean_columns", True)

        if use_mean_columns:
            y_values = ajuste["mean_f1"]
            y_error = ajuste["std_f1"]
        else:
            y_values = ajuste["f1_score"]
            y_error = None

        ax.plot(
            ajuste["lambda"],
            y_values,
            marker="o",
            linewidth=2,
            color=color,
        )
        if y_error is not None:
            ax.fill_between(
                ajuste["lambda"],
                y_values - y_error,
                y_values + y_error,
                color=color,
                alpha=0.12,
            )
        aplicar_marcas_log_espaciadas(ax, ajuste["lambda"])
        ax.set_title(label)
        ax.set_xlabel("lambda")
        ax.set_ylabel("F1")
        ax.grid(True, alpha=0.3)
        handle, = ax.plot([], [], color=color, marker="o", linewidth=2, label=label)
        legend_handles.append(handle)
        legend_labels.append(label)

    combined_ax = axes[3]
    for color, spec in zip(palette, tuning_specs):
        ajuste = spec["df"]
        use_mean_columns = spec.get("use_mean_columns", True)
        y_values = ajuste["mean_f1"] if use_mean_columns else ajuste["f1_score"]
        combined_ax.plot(
            ajuste["lambda"],
            y_values,
            marker="o",
            linewidth=2,
            color=color,
        )
    aplicar_marcas_log_espaciadas(combined_ax, tuning_specs[0]["df"]["lambda"])
    combined_ax.set_title("Comparacion conjunta")
    combined_ax.set_xlabel("lambda")
    combined_ax.set_ylabel("F1")
    combined_ax.grid(True, alpha=0.3)
    fig.legend(
        legend_handles,
        legend_labels,
        title="Estrategia",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
    )

    fig.suptitle("Validation F1 vs lambda", y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    return guardar_figura(fig, ruta_guardado)

def graficar_heatmap_matriz_confusion(matriz_confusion: pd.DataFrame, etiqueta_estrategia: str, ruta_guardado=None):
    return graficar_panel_matrices_confusion([{"df": matriz_confusion, "label": f"Matriz de confusion - {etiqueta_estrategia}"}], "", layout=(1, 1), figsize=(6, 5), ruta_guardado=ruta_guardado)

def graficar_triplete_matriz_confusion(confusion_specs, ruta_guardado=None):
    return graficar_panel_matrices_confusion(confusion_specs, "Matrices de confusion por estrategia", ruta_guardado=ruta_guardado)

def graficar_panel_curva_roc(roc_specs, ruta_guardado=None):
    return graficar_panel_curvas_binarias(roc_specs, "fpr", "tpr", "Curvas ROC por estrategia", "False Positive Rate", "True Positive Rate", legend_loc="lower right", ruta_guardado=ruta_guardado)

def graficar_panel_curva_precision_recall(pr_specs, ruta_guardado=None):
    return graficar_panel_curvas_binarias(pr_specs, "recall", "precision", "Curvas Precision-Recall por estrategia", "Recall", "Precision", baseline_key="baseline", legend_loc="lower left", ruta_guardado=ruta_guardado)

def graficar_comparacion_metricas_estrategias(
    metricas: pd.DataFrame,
    ruta_guardado=None,
):
    """Compara metricas escalares entre split strategies."""
    datos_grafico = metricas.melt(
        id_vars="strategy",
        value_vars=["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"],
        var_name="metric",
        value_name="value",
    )

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    sns.barplot(
        data=datos_grafico,
        x="metric",
        y="value",
        hue="strategy",
        ax=ax,
    )
    ax.set_title("Comparacion de metricas por estrategia de splitting")
    ax.set_xlabel("")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Estrategia", loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return guardar_figura(fig, ruta_guardado)

def graficar_conteos_rebalancing_example(
    balance_tabla: pd.DataFrame,
    ruta_guardado=None,
):
    """Compara conteos de clase antes y despues del rebalancing en un fold."""
    filas = []
    for fila in balance_tabla.to_dict("records"):
        filas.extend(
            [
                {
                    "technique_label": fila["technique_label"],
                    "stage": "Antes",
                    "class_label": "Insuficiente",
                    "count": fila["original_count_0"],
                },
                {
                    "technique_label": fila["technique_label"],
                    "stage": "Antes",
                    "class_label": "Aprobado",
                    "count": fila["original_count_1"],
                },
                {
                    "technique_label": fila["technique_label"],
                    "stage": "Despues",
                    "class_label": "Insuficiente",
                    "count": fila["fitted_count_0"],
                },
                {
                    "technique_label": fila["technique_label"],
                    "stage": "Despues",
                    "class_label": "Aprobado",
                    "count": fila["fitted_count_1"],
                },
            ]
        )
    melted_tabla = pd.DataFrame(filas)
    melted_tabla["stage_class"] = (
        melted_tabla["stage"] + " - " + melted_tabla["class_label"]
    )

    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.barplot(
        data=melted_tabla,
        y="technique_label",
        x="count",
        hue="stage_class",
        ax=ax,
    )
    ax.set_title("Como cambia el balance de clases segun el rebalancing method")
    ax.set_xlabel("Cantidad de observaciones")
    ax.set_ylabel("")
    ax.legend(title="Conteo", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    anotar_valores_barras(ax, orientation="horizontal", decimals=0)
    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_panel_lambda_rebalancing_binario(
    tuning_specs,
    ruta_guardado=None,
):
    """Muestra una curva de lambda por `rebalancing method` y una comparacion conjunta."""
    fig, axes = plt.subplots(2, 3, figsize=(17.2, 10.8))
    axes = axes.reshape(-1)
    palette = sns.color_palette("tab10", n_colors=max(len(tuning_specs), 3))
    legend_handles = []
    legend_labels = []

    for ax, color, spec in zip(axes[: len(tuning_specs)], palette, tuning_specs):
        ajuste = spec["df"].sort_values("lambda")
        ax.plot(
            ajuste["lambda"],
            ajuste["mean_f1"],
            marker="o",
            linewidth=2,
            color=color,
        )
        ax.fill_between(
            ajuste["lambda"],
            ajuste["mean_f1"] - ajuste["std_f1"],
            ajuste["mean_f1"] + ajuste["std_f1"],
            color=color,
            alpha=0.12,
        )
        best_idx = int(np.argmax(ajuste["mean_f1"].to_numpy()))
        best_lambda = float(ajuste.iloc[best_idx]["lambda"])
        best_f1 = float(ajuste.iloc[best_idx]["mean_f1"])
        ax.scatter([best_lambda], [best_f1], color="#bc4749", s=50, zorder=3)
        aplicar_marcas_log_espaciadas(ax, ajuste["lambda"])
        ax.set_title(spec["label"])
        ax.set_xlabel("lambda")
        ax.set_ylabel("F1")
        ax.grid(True, alpha=0.3)
        handle, = ax.plot([], [], color=color, marker="o", linewidth=2, label=spec["label"])
        legend_handles.append(handle)
        legend_labels.append(spec["label"])

    combined_ax = axes[len(tuning_specs)]
    for color, spec in zip(palette, tuning_specs):
        ajuste = spec["df"].sort_values("lambda")
        combined_ax.plot(
            ajuste["lambda"],
            ajuste["mean_f1"],
            marker="o",
            linewidth=2,
            color=color,
        )
    aplicar_marcas_log_espaciadas(
        combined_ax,
        tuning_specs[0]["df"].sort_values("lambda")["lambda"],
    )
    combined_ax.set_title("Comparacion conjunta")
    combined_ax.set_xlabel("lambda")
    combined_ax.set_ylabel("F1")
    combined_ax.grid(True, alpha=0.3)

    for ax in axes[len(tuning_specs) + 1 :]:
        ax.axis("off")

    fig.legend(
        legend_handles,
        legend_labels,
        title="Rebalancing method",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
    )
    fig.suptitle("Validation F1 vs lambda por rebalancing method", y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    return guardar_figura(fig, ruta_guardado)

def graficar_comparacion_metricas_rebalancing(
    metricas: pd.DataFrame,
    ruta_guardado=None,
):
    """Compara metricas binarias escalares entre rebalancing methods."""
    datos_grafico = metricas.melt(
        id_vars="technique_label",
        value_vars=["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"],
        var_name="metric",
        value_name="value",
    )
    orden_metricas = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    datos_grafico["metric"] = pd.Categorical(datos_grafico["metric"], categories=orden_metricas, ordered=True)

    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    sns.barplot(
        data=datos_grafico,
        y="metric",
        x="value",
        hue="technique_label",
        order=orden_metricas,
        ax=ax,
    )
    ax.set_title("Comparacion de metricas por rebalancing method")
    ax.set_xlabel("Valor")
    ax.set_ylabel("")
    ax.set_yticks(range(len(orden_metricas)))
    ax.set_yticklabels([etiqueta_metrica_multiclase(metrica) for metrica in orden_metricas])
    ax.set_xlim(0, 1.08)
    ax.legend(
        title="Rebalancing method",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        frameon=True,
    )
    anotar_valores_barras(ax, orientation="horizontal")
    fig.tight_layout(rect=[0, 0.16, 1, 0.96])
    return guardar_figura(fig, ruta_guardado)

def graficar_grilla_confusion_binaria(confusion_specs, ruta_guardado=None):
    return graficar_panel_matrices_confusion(confusion_specs, "Matrices de confusion por rebalancing method", layout=(2, 3), figsize=(15, 9), ruta_guardado=ruta_guardado)

def graficar_comparacion_roc_binaria(roc_specs, ruta_guardado=None):
    return graficar_curvas_superpuestas(roc_specs, "fpr", "tpr", "Curvas ROC por rebalancing method", "False Positive Rate", "True Positive Rate", legend_title="Rebalancing method", ruta_guardado=ruta_guardado)

def graficar_comparacion_pr_binaria(pr_specs, ruta_guardado=None):
    baseline = float(pr_specs[0]["baseline"]) if pr_specs else None
    return graficar_curvas_superpuestas(pr_specs, "recall", "precision", "Curvas Precision-Recall por rebalancing method", "Recall", "Precision", baseline=baseline, legend_title="Rebalancing method", ruta_guardado=ruta_guardado)

def graficar_triplete_evaluacion_binaria(
    matriz_confusion: pd.DataFrame,
    roc_spec: dict,
    pr_spec: dict,
    title: str = "Evaluacion binaria final en test",
    ruta_guardado=None,
):
    """Muestra matriz de confusion, curva ROC y curva PR en una sola fila."""
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.9))

    dibujar_matriz_confusion(axes[0], matriz_confusion, "Matriz de confusion")
    dibujar_curva(axes[1], roc_spec, "fpr", "tpr", "#2f6c8f")
    cerrar_eje_curva(axes[1], "Curva ROC", "False Positive Rate", "True Positive Rate", True, legend_loc="lower right")
    dibujar_curva(axes[2], pr_spec, "recall", "precision", "#bc4749")
    cerrar_eje_curva(
        axes[2],
        "Curva Precision-Recall",
        "Recall",
        "Precision",
        baseline=pr_spec["baseline"],
        legend_loc="lower left",
    )

    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_comparacion_std_coeficientes(
    stability_tabla: pd.DataFrame,
    cantidad_principales: int = 15,
    ruta_guardado=None,
):
    """Compara desvios de coeficientes por feature con segmentos horizontales."""
    datos_grafico = stability_tabla.head(cantidad_principales).copy()
    datos_grafico = datos_grafico.sort_values("std_gap_group_minus_random", ascending=True)
    positions = np.arange(len(datos_grafico))

    fig, ax = plt.subplots(figsize=(11, 7))
    for y_pos, (_, fila) in zip(positions, datos_grafico.iterrows()):
        ax.plot(
            [fila["coef_std_random"], fila["coef_std_group"]],
            [y_pos, y_pos],
            color="#b0b0b0",
            linewidth=2,
        )

    ax.scatter(
        datos_grafico["coef_std_random"],
        positions,
        color="#2f6c8f",
        label="Random KFold",
        s=55,
        zorder=3,
    )
    ax.scatter(
        datos_grafico["coef_std_group"],
        positions,
        color="#bc4749",
        label="GroupKFold",
        s=55,
        zorder=3,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(datos_grafico["feature"])
    ax.set_title("Desvio estandar de coeficientes en las features mas inestables")
    ax.set_xlabel("Desvio estandar del coeficiente")
    ax.set_ylabel("Feature")
    ax.legend(loc="best")
    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_distribuciones_coeficientes(
    coeficientes: pd.DataFrame,
    features_seleccionadas,
    ruta_guardado=None,
):
    """Grafica distribuciones de coeficientes por fold para algunas features."""
    datos_grafico = coeficientes[coeficientes["feature"].isin(features_seleccionadas)].copy()

    fig, ax = plt.subplots(figsize=(16, 6.5))
    sns.boxplot(
        data=datos_grafico,
        x="feature",
        y="coefficient",
        hue="strategy",
        ax=ax,
    )
    ax.set_title("Distribucion de coeficientes por estrategia de CV")
    ax.set_xlabel("")
    ax.set_ylabel("Coeficiente")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="CV", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_ajuste_lambda_multiclase(ajuste: pd.DataFrame, ruta_guardado=None):
    return graficar_curva_ajuste_regularizacion(ajuste, x_col="lambda_", y_col="mean_macro_f1", error_col="std_macro_f1", title="Macro Validation F1 vs lambda - logistica multiclase", x_label="lambda", color="#2f6c8f", ruta_guardado=ruta_guardado)

def graficar_ajuste_ridge_lda(ajuste: pd.DataFrame, ruta_guardado=None):
    return graficar_curva_ajuste_regularizacion(ajuste, x_col="covariance_ridge", y_col="mean_macro_f1", error_col="std_macro_f1", title="Macro Validation F1 vs ridge - LDA", x_label="covariance_ridge", color="#9c6644", ruta_guardado=ruta_guardado)

def graficar_ajuste_random_forest(
    ajuste: pd.DataFrame,
    ruta_guardado=None,
):
    """Muestra macro F1 con GroupKFold para configuraciones de Random Forest."""
    datos_grafico = ajuste.copy().sort_values("mean_macro_f1", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(datos_grafico["config_label"], datos_grafico["mean_macro_f1"], color="#8da9c4")
    ax.set_title("Macro Validation F1 por configuracion del Random Forest")
    ax.set_xlabel("Macro F1 medio en GroupKFold")
    ax.set_ylabel("Configuracion")
    ax.set_xlim(0, min(1.06, datos_grafico["mean_macro_f1"].max() * 1.18))
    anotar_valores_barras(ax, orientation="horizontal")

    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_comparacion_metricas_modelos_multiclase(
    metricas: pd.DataFrame,
    ruta_guardado=None,
):
    """Compara metricas macro y accuracy global entre modelos multiclase."""
    orden_metricas = [
        "overall_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_roc_auc",
        "macro_pr_auc",
    ]
    datos_grafico = metricas.melt(
        id_vars="model",
        value_vars=orden_metricas,
        var_name="metric",
        value_name="value",
    )
    datos_grafico["metric"] = pd.Categorical(datos_grafico["metric"], categories=orden_metricas, ordered=True)

    fig, ax = plt.subplots(figsize=(12.6, 7.0))
    sns.barplot(data=datos_grafico, y="metric", x="value", hue="model", order=orden_metricas, ax=ax)
    ax.set_title("Comparacion global de metricas multiclase")
    ax.set_xlabel("Valor")
    ax.set_ylabel("")
    ax.set_yticks(range(len(orden_metricas)))
    ax.set_yticklabels([etiqueta_metrica_multiclase(metrica) for metrica in orden_metricas])
    ax.set_xlim(0, 1.08)
    ax.legend(title="Modelo", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    anotar_valores_barras(ax, orientation="horizontal")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return guardar_figura(fig, ruta_guardado)

def graficar_triplete_confusion_multiclase(confusion_specs, ruta_guardado=None):
    return graficar_panel_matrices_confusion(confusion_specs, "Matrices de confusion multiclase por modelo", layout=(1, 3), figsize=(17, 4.8), ruta_guardado=ruta_guardado)

def graficar_panel_metricas_clase_multiclase(
    metricas_por_clase: pd.DataFrame,
    metricas=None,
    ruta_guardado=None,
):
    """Muestra metricas one-vs-all por clase en un panel 2x2."""
    metricas = metricas or ["f1_score", "accuracy", "precision", "recall"]
    orden_clases = [
        label for label in ORDEN_OBJETIVO_MULTICLASE if label in metricas_por_clase["class"].unique()
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.reshape(-1)

    legend_handles = None
    legend_labels = None

    for ax, metrica in zip(axes, metricas):
        sns.barplot(
            data=metricas_por_clase,
            y="class",
            x=metrica,
            hue="model",
            order=orden_clases,
            ax=ax,
        )
        ax.set_title(etiqueta_metrica_multiclase(metrica))
        ax.set_xlabel(etiqueta_metrica_multiclase(metrica))
        ax.set_ylabel("Clase")
        ax.set_xlim(0, 1.08)
        anotar_valores_barras(ax, orientation="horizontal")

        legend = ax.get_legend()
        if legend is not None:
            if legend_handles is None:
                legend_handles = legend.legend_handles
                legend_labels = [text.get_text() for text in legend.get_texts()]
            legend.remove()

    for ax in axes[len(metricas):]:
        ax.axis("off")

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Modelo",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
        )

    fig.suptitle("Metricas one-vs-all por clase", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return guardar_figura(fig, ruta_guardado)

def graficar_panel_roc_one_vs_all(
    especificaciones_modelos,
    etiquetas_clases=None,
    ruta_guardado=None,
):
    """Compara curvas ROC one-vs-all por clase en un panel 2x2."""
    etiquetas_clases = etiquetas_clases or ORDEN_OBJETIVO_MULTICLASE
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.8))
    axes = axes.reshape(-1)

    palette = ["#2f6c8f", "#bc4749", "#4c956c"]
    legend_handles = []

    for indice_clase, (ax, etiqueta_clase) in enumerate(zip(axes, etiquetas_clases)):
        summary_lines = []
        for color, spec in zip(palette, especificaciones_modelos):
            curve_tabla = spec["roc_curves"][etiqueta_clase]
            line, = ax.plot(curve_tabla["fpr"], curve_tabla["tpr"], linewidth=2, color=color, label=spec["label"])
            if indice_clase == 0:
                legend_handles.append(line)
            summary_lines.append(f"{spec['label']}: {valor_metrica_clase(spec, etiqueta_clase, 'roc_auc'):.3f}")

        cerrar_eje_curva(
            ax,
            f"ROC one-vs-all - {etiqueta_clase}",
            "False Positive Rate",
            "True Positive Rate",
            diagonal=True,
            legend_loc=None,
        )
        anotar_resumen_curvas(ax, "AUC", summary_lines)

    if legend_handles:
        fig.legend(
            legend_handles,
            [spec["label"] for spec in especificaciones_modelos],
            title="Modelo",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return guardar_figura(fig, ruta_guardado)

def graficar_panel_pr_one_vs_all(
    especificaciones_modelos,
    etiquetas_clases=None,
    ruta_guardado=None,
):
    """Compara curvas PR one-vs-all por clase en un panel 2x2."""
    etiquetas_clases = etiquetas_clases or ORDEN_OBJETIVO_MULTICLASE
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.8))
    axes = axes.reshape(-1)

    palette = ["#2f6c8f", "#bc4749", "#4c956c"]
    legend_handles = []

    for indice_clase, (ax, etiqueta_clase) in enumerate(zip(axes, etiquetas_clases)):
        baselines = []
        summary_lines = []
        for color, spec in zip(palette, especificaciones_modelos):
            curve_tabla = spec["pr_curves"][etiqueta_clase]
            linea_base = float(spec["positive_rates"][etiqueta_clase])
            baselines.append(linea_base)
            line, = ax.plot(curve_tabla["recall"], curve_tabla["precision"], linewidth=2, color=color, label=spec["label"])
            if indice_clase == 0:
                legend_handles.append(line)
            summary_lines.append(f"{spec['label']}: {valor_metrica_clase(spec, etiqueta_clase, 'pr_auc'):.3f}")

        baseline_mean = float(np.mean(baselines))
        cerrar_eje_curva(
            ax,
            f"PR one-vs-all - {etiqueta_clase}",
            "Recall",
            "Precision",
            baseline=baseline_mean,
            legend_loc=None,
        )
        anotar_resumen_curvas(ax, "PR-AUC", summary_lines + [f"base: {baseline_mean:.3f}"])

    if legend_handles:
        fig.legend(
            legend_handles,
            [spec["label"] for spec in especificaciones_modelos],
            title="Modelo",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return guardar_figura(fig, ruta_guardado)

def graficar_principales_importancias_features(
    importance_tabla: pd.DataFrame,
    etiqueta_modelo: str = "Random Forest",
    cantidad_principales: int = 15,
    ruta_guardado=None,
):
    """Muestra las features mas importantes del Random Forest."""
    datos_grafico = (
        importance_tabla.sort_values("mean_importance", ascending=True)
        .tail(cantidad_principales)
        .copy()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(datos_grafico["feature"], datos_grafico["mean_importance"], color="#4c956c")
    ax.set_title(f"Importancia de features - {etiqueta_modelo}")
    ax.set_xlabel("Importancia media")
    ax.set_ylabel("Feature")
    ax.set_xlim(0, datos_grafico["mean_importance"].max() * 1.18)
    anotar_valores_barras(ax, orientation="horizontal")
    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)

def graficar_estimacion_curva_aprendizaje_rf(
    resumen_curva: pd.DataFrame,
    curva_ajustada: pd.DataFrame,
    estimacion: pd.DataFrame,
    ruta_guardado=None,
):
    """Visualiza la curva de aprendizaje del Random Forest y el objetivo de 1%."""
    estimate_row = estimacion.iloc[0]
    current_accuracy = float(estimate_row["current_accuracy"])
    target_accuracy = float(estimate_row["target_accuracy"])
    current_n = float(estimate_row["current_n"])
    required_n = estimate_row["required_n"]
    finite_solution = bool(estimate_row["finite_solution"])
    fit_method = str(estimate_row.get("fit_method", "ajuste lineal local"))
    fit_points = int(estimate_row.get("fit_points", 3))

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.errorbar(
        resumen_curva["n_samples"], resumen_curva["mean_accuracy"],
        yerr=resumen_curva["std_accuracy"], fmt="o", color="#2f6c8f",
        ecolor="#9ecae1", elinewidth=1.5, capsize=4, label="mediciones GroupKFold",
    )
    ax.plot(
        curva_ajustada["n_samples"], curva_ajustada["predicted_accuracy"],
        color="#bc4749", linewidth=2.2, label=f"{fit_method} ({fit_points} puntos)",
    )
    for valor, color, etiqueta in [
        (current_accuracy, "#6c757d", f"accuracy actual = {current_accuracy:.3f}"),
        (target_accuracy, "#4c956c", f"objetivo = {target_accuracy:.3f}"),
    ]:
        ax.axhline(valor, color=color, linestyle="--", linewidth=1.4, label=etiqueta)
    ax.axvline(current_n, color="#6c757d", linestyle=":", linewidth=1.3)

    if finite_solution:
        ax.axvline(required_n, color="#4c956c", linestyle=":", linewidth=1.3)
        texto = f"n requerido ~ {required_n:.0f}\nextra ~ {int(estimate_row['additional_n'])}"
        xy, xytext, color_borde = (required_n, target_accuracy), (10, -18), "#4c956c"
    else:
        texto = "con este ajuste no aparece\nsolucion finita para +1%"
        xy, xytext, color_borde = (current_n, target_accuracy), (12, 12), "#bc4749"
    ax.annotate(
        texto, xy=xy, xytext=xytext, textcoords="offset points", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color_borde, alpha=0.9),
    )

    ax.set_title("Curva de aprendizaje del Random Forest multiclase")
    ax.set_xlabel("Cantidad de observaciones de desarrollo")
    ax.set_ylabel("Accuracy global")
    ax.set_ylim(
        min(resumen_curva["mean_accuracy"].min() - 0.02, current_accuracy - 0.02),
        max(target_accuracy + 0.02, resumen_curva["mean_accuracy"].max() + 0.02),
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return guardar_figura(fig, ruta_guardado)
