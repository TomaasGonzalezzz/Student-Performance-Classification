from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_DATOS = RAIZ_PROYECTO / "data"
CARPETA_DATOS_CRUDOS = CARPETA_DATOS / "raw"
CARPETA_DATOS_PROCESADOS = CARPETA_DATOS / "processed"
CARPETA_DOCS = RAIZ_PROYECTO / "docs"
CARPETA_NOTEBOOK = RAIZ_PROYECTO / "notebook"
CARPETA_SALIDAS = RAIZ_PROYECTO / "outputs"
CARPETA_FIGURAS = CARPETA_SALIDAS / "figures"
CARPETA_TABLAS = CARPETA_SALIDAS / "tables"
CARPETA_FIGURAS_PUNTO_1 = CARPETA_FIGURAS / "punto_1"
CARPETA_TABLAS_PUNTO_1 = CARPETA_TABLAS / "punto_1"
CARPETA_FIGURAS_PUNTO_2 = CARPETA_FIGURAS / "punto_2"
CARPETA_TABLAS_PUNTO_2 = CARPETA_TABLAS / "punto_2"
CARPETA_FIGURAS_PUNTO_3 = CARPETA_FIGURAS / "punto_3"
CARPETA_TABLAS_PUNTO_3 = CARPETA_TABLAS / "punto_3"
CARPETA_FIGURAS_PUNTO_4 = CARPETA_FIGURAS / "punto_4"
CARPETA_TABLAS_PUNTO_4 = CARPETA_TABLAS / "punto_4"
CARPETA_FIGURAS_PUNTO_5 = CARPETA_FIGURAS / "punto_5"
CARPETA_TABLAS_PUNTO_5 = CARPETA_TABLAS / "punto_5"

FEATURES_NUMERICAS = [
    "horas_estudio",
    "asistencia",
    "nota_previa",
    "horas_sueno",
    "participacion",
    "horas_extracurricular",
    "distancia_escuela_km",
    "nivel_socioeconomico",
    "tamano_clase",
]

FEATURES_BINARIAS = ["acceso_internet"]

FEATURES_CATEGORICAS = ["escuela", "semestre"]

FEATURES_MODELO = FEATURES_NUMERICAS + FEATURES_BINARIAS + FEATURES_CATEGORICAS

ORDEN_ESCUELAS = list("ABCDEFGH")
ORDEN_SEMESTRES = [
    "2022-1",
    "2022-2",
    "2023-1",
    "2023-2",
    "2024-1",
    "2024-2",
    "2025-1",
]

ORDEN_OBJETIVO_MULTICLASE = ["Insuficiente", "Regular", "Bueno", "Excelente"]
ORDEN_OBJETIVO_BINARIO = ["Insuficiente", "Aprobado"]
MAPA_OBJETIVO_ORDINAL = {
    "Insuficiente": 0,
    "Regular": 1,
    "Bueno": 2,
    "Excelente": 3,
}

EXPORTAR_FIGURAS = False
EXPORTAR_TABLAS = False

ESTILO_SEABORN = {
    "style": "whitegrid",
    "context": "talk",
}
