from fastapi import FastAPI
import pandas as pd
import numpy as np
from datetime import datetime
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

app = FastAPI(title="Pronóstico de Dengue – Chincha")

# =========================
# MODEL INITIALIZATION
# =========================

def build_forecast():
    df = pd.read_csv("dengue_ica_chincha.csv")
    df.columns = df.columns.str.lower()
    df = df[df["enfermedad"].str.contains("dengue", case=False, na=False)]

    df_agg = (
        df.groupby(["ubigeo", "distrito", "ano", "semana"])
        .size()
        .reset_index(name="casos")
        .sort_values(["ubigeo", "ano", "semana"])
        .reset_index(drop=True)
    )

    le = LabelEncoder()
    df_agg["ubigeo_id"] = le.fit_transform(df_agg["ubigeo"])

    df_agg["semana_sin"] = np.sin(2 * np.pi * df_agg["semana"] / 52)
    df_agg["semana_cos"] = np.cos(2 * np.pi * df_agg["semana"] / 52)

    for lag in [1, 2, 4, 8, 12]:
        df_agg[f"lag_{lag}"] = df_agg.groupby("ubigeo")["casos"].shift(lag)

    df_agg["ma_4"] = (
        df_agg.groupby("ubigeo")["casos"]
        .shift(1)
        .rolling(4)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df_agg["ma_8"] = (
        df_agg.groupby("ubigeo")["casos"]
        .shift(1)
        .rolling(8)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df_model = df_agg.dropna()

    features = [
        "ano", "semana", "ubigeo_id",
        "semana_sin", "semana_cos",
        "lag_1", "lag_2", "lag_4", "lag_8", "lag_12",
        "ma_4", "ma_8"
    ]

    X = df_model[features]
    y = df_model["casos"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="count:poisson",
        random_state=42
    )

    model.fit(X, y)

    # ========= FORECAST =========
    def forecast_future(years=3):
        future = df_model.copy()
        last_year = df_model["ano"].max()
        results = []

        for u in future["ubigeo"].unique():
            dist_data = future[future["ubigeo"] == u].tail(12).copy()

            for y in range(1, years + 1):
                for w in range(1, 53):
                    recent = dist_data.tail(12)

                    row = {
                        "ubigeo": u,
                        "distrito": recent.iloc[-1]["distrito"],
                        "ano": last_year + y,
                        "semana": w,
                        "ubigeo_id": recent.iloc[-1]["ubigeo_id"],
                        "semana_sin": np.sin(2 * np.pi * w / 52),
                        "semana_cos": np.cos(2 * np.pi * w / 52),
                        "lag_1": recent.iloc[-1]["casos"],
                        "lag_2": recent.iloc[-2]["casos"],
                        "lag_4": recent.iloc[-4]["casos"],
                        "lag_8": recent.iloc[-8]["casos"],
                        "lag_12": recent.iloc[-12]["casos"],
                        "ma_4": recent.tail(4)["casos"].mean(),
                        "ma_8": recent.tail(8)["casos"].mean(),
                    }

                    pred = int(round(max(0, model.predict(pd.DataFrame([row])[features])[0])))
                    row["casos"] = pred
                    dist_data = pd.concat([dist_data, pd.DataFrame([row])], ignore_index=True)
                    results.append(row)

        return pd.DataFrame(results)

    forecast = forecast_future()

    historical = df_agg[df_agg["ano"] >= df_agg["ano"].max() - 1][
        ["ubigeo", "distrito", "ano", "semana", "casos"]
    ]

    all_data = pd.concat([historical, forecast], ignore_index=True)

    return {
        "metadatos": {
            "generado_en": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "modelo": "XGBoost",
            "anos_pronosticados": 3
        },
        "datos": all_data.rename(columns={
            "ano": "año",
            "casos": "nro_casos"
        }).to_dict(orient="records")
    }


# =========================
# LOAD ON STARTUP
# =========================
@app.on_event("startup")
def load_model():
    global RESULT_JSON
    RESULT_JSON = build_forecast()


# =========================
# API ENDPOINT
# =========================
@app.get("/forecast")
def get_forecast():
    return RESULT_JSON
