from pathlib import Path

import pandas as pd

from configuracion import EXPORTAR_TABLAS


def asegurar_directorio(ruta):
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def preparar_ruta_salida(ruta):
    ruta = Path(ruta)
    asegurar_directorio(ruta.parent)
    return ruta


def guardar_dataframe(datos, ruta, index=True, habilitado=None):
    habilitado = EXPORTAR_TABLAS if habilitado is None else bool(habilitado)
    if not habilitado:
        return Path(ruta)

    ruta = preparar_ruta_salida(ruta)
    serializable = datos.copy()

    for col in serializable.columns:
        if pd.api.types.is_categorical_dtype(serializable[col]):
            serializable[col] = serializable[col].astype(str)

    serializable.to_csv(ruta, index=index)
    return ruta

