import os
import logging
import sqlite3
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# --- Logging -----------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
log = logging.getLogger(__name__)

# --- Configuração via variáveis de ambiente ----------------------------------
DB_PATH    = os.getenv("DB_PATH",   "data/heat_exchanger.db")
MODEL_DIR  = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


def load_data(db_path: str) -> pd.DataFrame:
    log.info("Conectando ao banco: %s", db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp, heat_efficiency FROM heat_exchanger ORDER BY timestamp",
        conn,
    )
    conn.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day_index"] = (df["timestamp"] - df["timestamp"].min()).dt.days
    log.info("Dados carregados: %d registros | período: %s → %s",
             len(df), df["timestamp"].min().date(), df["timestamp"].max().date())
    return df


def train(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    log.info("Iniciando treino — modelo: LinearRegression")
    model = LinearRegression()
    model.fit(X, y)
    log.info("Treino concluído — coef=%.6f  intercept=%.4f", model.coef_[0], model.intercept_)
    return model


def evaluate(model: LinearRegression, X: np.ndarray, y: np.ndarray):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    y_pred = model.predict(X)

    log.info("MAE=%.4f%%  RMSE=%.4f%%  R²=%.4f  R²_cv=%.4f±%.4f  tendência=%.4f%%/dia",
             mean_absolute_error(y, y_pred),
             root_mean_squared_error(y, y_pred),
             r2_score(y, y_pred),
             cv_scores.mean(), cv_scores.std(),
             model.coef_[0])


def save_artifacts(model: LinearRegression):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    log.info("Modelo salvo: %s", MODEL_PATH)


if __name__ == "__main__":
    df = load_data(DB_PATH)
    log.info("Eficiência: min=%.2f%%  max=%.2f%%", df["heat_efficiency"].min(), df["heat_efficiency"].max())

    X = df[["day_index"]].values
    y = df["heat_efficiency"].values

    model = train(X, y)
    evaluate(model, X, y)
    save_artifacts(model)