"""
inference.py — dois modos de uso:

  Modo 1 — prever eficiência para uma data:
    python src/inference.py --date 2022-04-15

  Modo 2 — estimar a data para uma eficiência alvo:
    python src/inference.py --efficiency 94.5
    python src/inference.py --efficiency 82      (extrapolação futura)
"""
import os
import logging
import sqlite3
import argparse
import pickle
import pandas as pd

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


def load_artifacts():
    log.info("Carregando modelo: %s", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo não encontrado em: {MODEL_PATH}\n"
            "Execute 'python src/train.py' primeiro."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    log.info("Carregando dados históricos: %s", DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT timestamp, heat_efficiency FROM heat_exchanger ORDER BY timestamp",
        conn,
    )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day_index"] = (df["timestamp"] - df["timestamp"].min()).dt.days

    log.info("Artefatos prontos — %d registros históricos carregados", len(df))
    return model, df


def predict_efficiency(model, df: pd.DataFrame, date_str: str) -> dict:
    target_date = pd.to_datetime(date_str)
    origin      = df["timestamp"].min()
    day_index   = (target_date - origin).days

    if day_index < 0:
        raise ValueError(f"Data anterior ao início do dataset ({origin.date()}).")

    efficiency = model.predict([[day_index]])[0]
    closest    = df.iloc[(df["day_index"] - day_index).abs().argsort()[:1]]

    log.info("Predição: data=%s  day_index=%d  eficiência=%.4f%%", date_str, day_index, efficiency)
    log.info("Referência histórica: data=%s  eficiência=%.4f%%",
             closest["timestamp"].iloc[0].date(), closest["heat_efficiency"].iloc[0])

    return {
        "input_date": date_str,
        "predicted_efficiency": round(float(efficiency), 4),
        "closest_historical_date": str(closest["timestamp"].iloc[0].date()),
        "closest_historical_efficiency": round(float(closest["heat_efficiency"].iloc[0]), 4),
    }


def find_date_for_efficiency(model, df: pd.DataFrame, target_efficiency: float, top_k: int = 3) -> dict:
    # Regressão inversa: y = a·x + b  →  x = (y − b) / a
    a              = model.coef_[0]
    b              = model.intercept_
    predicted_day  = (target_efficiency - b) / a
    origin         = df["timestamp"].min()
    predicted_date = origin + pd.Timedelta(days=round(predicted_day))
    in_history     = predicted_date.date() <= df["timestamp"].max().date()

    log.info("Regressão inversa: eficiência=%.2f%%  dia_previsto=%.1f  data=%s  [%s]",
             target_efficiency, predicted_day, predicted_date.date(),
             "histórico" if in_history else "extrapolação futura")

    df = df.copy()
    df["diff"] = (df["heat_efficiency"] - target_efficiency).abs()
    historical = df.nsmallest(top_k, "diff")

    for i, row in enumerate(historical.itertuples(), 1):
        log.info("  match %d: data=%s  eficiência=%.4f%%  delta=%.4f%%",
                 i, row.timestamp.date(), row.heat_efficiency, row.diff)

    return {
        "target_efficiency": target_efficiency,
        "predicted_date": str(predicted_date.date()),
        "in_history": in_history,
        "historical_matches": [
            {
                "date": str(row["timestamp"].date()),
                "recorded_efficiency": round(float(row["heat_efficiency"]), 4),
                "delta": round(float(row["diff"]), 4),
            }
            for _, row in historical.iterrows()
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferência — Trocador de Calor")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date",       type=str,   help="Data para prever eficiência (YYYY-MM-DD)")
    group.add_argument("--efficiency", type=float, help="Eficiência alvo para estimar a data")
    parser.add_argument("--top",       type=int, default=3, help="Número de registros históricos")
    args = parser.parse_args()

    model, df = load_artifacts()

    if args.date:
        predict_efficiency(model, df, args.date)
    else:
        find_date_for_efficiency(model, df, args.efficiency, top_k=args.top)
