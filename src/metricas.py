from __future__ import annotations

import numpy as np
import pandas as pd


def metricas_binarias_desde_etiquetas(y_true, y_pred) -> dict:
    matriz_confusion = matriz_confusion_binaria(y_true, y_pred)
    tn = matriz_confusion.loc["real_0", "pred_0"]
    fp = matriz_confusion.loc["real_0", "pred_1"]
    fn = matriz_confusion.loc["real_1", "pred_0"]
    tp = matriz_confusion.loc["real_1", "pred_1"]

    accuracy = (tp + tn) / max(len(np.asarray(y_true)), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "confusion_matrix": matriz_confusion,
    }


def matriz_confusion_binaria(y_true, y_pred) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))

    return pd.DataFrame(
        [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        index=["real_0", "real_1"],
        columns=["pred_0", "pred_1"],
    )


def auc_trapecio(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.trapezoid(y, x))


def curva_roc_binaria(y_true, puntajes) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    puntajes = np.asarray(puntajes, dtype=float)

    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())

    if positives == 0 or negatives == 0:
        return pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})

    order = np.argsort(-puntajes, kind="mergesort")
    sorted_y = y_true[order]
    sorted_scores = puntajes[order]

    cumulative_tp = np.cumsum(sorted_y == 1)
    cumulative_fp = np.cumsum(sorted_y == 0)
    distinct_score_idx = np.where(np.diff(sorted_scores))[0]
    threshold_idx = np.r_[distinct_score_idx, len(sorted_scores) - 1]

    tpr = np.r_[0.0, cumulative_tp[threshold_idx] / positives, 1.0]
    fpr = np.r_[0.0, cumulative_fp[threshold_idx] / negatives, 1.0]

    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def curva_precision_recall_binaria(y_true, puntajes) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    puntajes = np.asarray(puntajes, dtype=float)

    positives = int(y_true.sum())
    if positives == 0:
        return pd.DataFrame({"recall": [0.0, 1.0], "precision": [1.0, 1.0]})

    order = np.argsort(-puntajes, kind="mergesort")
    sorted_y = y_true[order]
    sorted_scores = puntajes[order]

    cumulative_tp = np.cumsum(sorted_y == 1)
    cumulative_fp = np.cumsum(sorted_y == 0)
    distinct_score_idx = np.where(np.diff(sorted_scores))[0]
    threshold_idx = np.r_[distinct_score_idx, len(sorted_scores) - 1]

    precision = cumulative_tp[threshold_idx] / (
        cumulative_tp[threshold_idx] + cumulative_fp[threshold_idx]
    )
    recall = cumulative_tp[threshold_idx] / positives

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    return pd.DataFrame({"recall": recall, "precision": precision})


def reporte_clasificacion_binaria(
    y_true,
    puntajes,
    umbral: float = 0.5,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    puntajes = np.asarray(puntajes, dtype=float)
    y_pred = (puntajes >= umbral).astype(int)

    scalar_metrics = metricas_binarias_desde_etiquetas(y_true, y_pred)
    curva_roc = curva_roc_binaria(y_true, puntajes)
    curva_pr = curva_precision_recall_binaria(y_true, puntajes)

    roc_auc = auc_trapecio(curva_roc["fpr"], curva_roc["tpr"])
    pr_auc = auc_trapecio(curva_pr["recall"], curva_pr["precision"])

    metricas = scalar_metrics.copy()
    matriz_confusion = metricas.pop("confusion_matrix")
    metricas["roc_auc"] = float(roc_auc)
    metricas["pr_auc"] = float(pr_auc)

    return {
        "metrics": metricas,
        "confusion_matrix": matriz_confusion,
        "roc_curve": curva_roc,
        "pr_curve": curva_pr,
        "y_pred": y_pred,
    }


def matriz_confusion_multiclase(
    y_true,
    y_pred,
    etiquetas_clases,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cantidad_clases = len(etiquetas_clases)
    matrix = np.zeros((cantidad_clases, cantidad_clases), dtype=int)

    for true_idx, pred_idx in zip(y_true, y_pred):
        matrix[int(true_idx), int(pred_idx)] += 1

    return pd.DataFrame(
        matrix,
        index=[f"real_{label}" for label in etiquetas_clases],
        columns=[f"pred_{label}" for label in etiquetas_clases],
    )


def reporte_multiclase_one_vs_all(
    y_true,
    puntajes,
    etiquetas_clases,
) -> dict:
    """Evalua un modelo multiclase en formato one-vs-all.

    La clase predicha sale del argmax sobre los scores o probabilidades.
    Las curvas ROC y PR se construyen tratando a cada clase como positiva
    frente al resto.
    """
    y_true = np.asarray(y_true, dtype=int)
    puntajes = np.asarray(puntajes, dtype=float)
    y_pred = np.argmax(puntajes, axis=1).astype(int)

    matriz_confusion = matriz_confusion_multiclase(y_true, y_pred, etiquetas_clases)
    overall_accuracy = float(np.mean(y_pred == y_true))

    per_class_rows = []
    roc_curves = {}
    pr_curves = {}
    positive_rates = {}

    for indice_clase, etiqueta_clase in enumerate(etiquetas_clases):
        class_true = (y_true == indice_clase).astype(int)
        class_pred = (y_pred == indice_clase).astype(int)
        puntajes_clase = puntajes[:, indice_clase]

        scalar_metrics = metricas_binarias_desde_etiquetas(class_true, class_pred)
        curva_roc = curva_roc_binaria(class_true, puntajes_clase)
        curva_pr = curva_precision_recall_binaria(class_true, puntajes_clase)
        roc_auc = auc_trapecio(curva_roc["fpr"], curva_roc["tpr"])
        pr_auc = auc_trapecio(curva_pr["recall"], curva_pr["precision"])

        per_class_rows.append(
            {
                "class": etiqueta_clase,
                "support": int(class_true.sum()),
                "accuracy": scalar_metrics["accuracy"],
                "precision": scalar_metrics["precision"],
                "recall": scalar_metrics["recall"],
                "f1_score": scalar_metrics["f1_score"],
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
            }
        )
        roc_curves[etiqueta_clase] = curva_roc
        pr_curves[etiqueta_clase] = curva_pr
        positive_rates[etiqueta_clase] = float(class_true.mean())

    metricas_por_clase = pd.DataFrame(per_class_rows)
    active_class_tabla = metricas_por_clase[metricas_por_clase["support"] > 0].copy()
    if active_class_tabla.empty:
        active_class_tabla = metricas_por_clase.copy()

    macro_metrics = {
        "overall_accuracy": overall_accuracy,
        "macro_accuracy": float(active_class_tabla["accuracy"].mean()),
        "macro_precision": float(active_class_tabla["precision"].mean()),
        "macro_recall": float(active_class_tabla["recall"].mean()),
        "macro_f1": float(active_class_tabla["f1_score"].mean()),
        "macro_roc_auc": float(active_class_tabla["roc_auc"].mean()),
        "macro_pr_auc": float(active_class_tabla["pr_auc"].mean()),
    }

    return {
        "metrics": macro_metrics,
        "per_class_metrics": metricas_por_clase,
        "confusion_matrix": matriz_confusion,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
        "positive_rates": positive_rates,
        "y_pred": y_pred,
    }
