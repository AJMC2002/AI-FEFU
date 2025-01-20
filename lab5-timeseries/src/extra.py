import os
from datetime import datetime
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

DB_HOST = os.getenv("DB_HOST","db")
DB_USER = os.getenv("POSTGRES_USER","user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD","password")
DB_NAME = os.getenv("POSTGRES_DB", "intensitydb")
DB_PORT = os.getenv("DB_PORT", 5435)

def fetch_weather_data():
    weather_data = []
    for day in [
        f"20{yy:02d}-{mm:02d}-{dd:02d}"
        for yy in range(18, 25)
        for mm in range(1, 13)
        for dd in range(1, 29, 7)
    ]:
        headers = {"Accept": "application/json"}
        response = requests.get(
            f"https://api.carbonintensity.org.uk/intensity/date/{day}/{25}",
            params={},
            headers=headers,
        )
        data = response.json()

        if response.status_code == 200 and len(data["data"]) > 0:
            dt = datetime.strptime(data["data"][0]["from"], "%Y-%m-%dT%H:%MZ")
            intensity = data["data"][0]["intensity"]["actual"]
            weather_data.append(
                {
                    "date": dt,
                    "intensity": intensity,
                }
            )
            print(f"[EXTRA TASK] API RETURNED 1 ITEM.")

        sleep(0.5)  # Prevent getting ip-blocked lol

    print(f"[EXTRA TASK] API DATA REQUESTED. OBTAINED {len(weather_data)} ITEMS.")

    return weather_data


def insert_data_to_db(weather_data):
    conn = psycopg2.connect(
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            host=DB_HOST,
            )
    cursor = conn.cursor()

    print("[EXTRA TASK] CONNECTED TO DB. CREATING TABLE.")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS intensity (
            id SERIAL PRIMARY KEY,
            date TIMESTAMP,
            intensity INT,
            UNIQUE(date)
        );
    """)

    print("[EXTRA TASK] INSERTING VALUES.")

    for entry in weather_data:
        cursor.execute(
            """
            INSERT INTO intensity (date, intensity)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING;
        """,
            (entry["date"], entry["intensity"]),
        )

    conn.commit()
    cursor.close()
    conn.close()


def retrieve_data_from_db():
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST
    )
    cursor = conn.cursor()

    print("[EXTRA TASK] EXTRACTING DATA.")

    cursor.execute("SELECT date, intensity FROM intensity")
    rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=["date", "intensity"])

    cursor.close()
    conn.close()
    return df


def main():
    print("[EXTRA TASK] Starting...")

    weather_data = fetch_weather_data()

    insert_data_to_db(weather_data)

    df = retrieve_data_from_db()
    df["year"] = pd.Series.apply(df["date"], lambda date: date.year)
    df["month"] = pd.Series.apply(df["date"], lambda date: date.month)
    df["day"] = pd.Series.apply(df["date"], lambda date: date.day)
    df = df.dropna().sort_values("date")

    print(df)

    X = df[["year", "month", "day"]]
    y = df["intensity"]

    model = RandomForestRegressor(n_estimators=200, random_state=19)
    model.fit(X, y)

    df_test = df[df["year"] == 2018]
    X_test = df_test[["year", "month", "day"]]
    y_test = df_test["intensity"]
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"[EXTRA TASK] Mean Absolute Error: {mae}")
    print(f"[EXTRA TASK] Root Mean Squared Error: {rmse}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(
        df_test["date"],
        min(y_test),
        max(y_test),
        where=pd.Series.apply(df_test["month"], lambda month: month in range(3, 6)),
        color="b",
        alpha=0.1,
    )
    ax.fill_between(
        df_test["date"],
        min(y_test),
        max(y_test),
        where=pd.Series.apply(df_test["month"], lambda month: month in range(6, 9)),
        color="g",
        alpha=0.1,
    )
    ax.fill_between(
        df_test["date"],
        min(y_test),
        max(y_test),
        where=pd.Series.apply(df_test["month"], lambda month: month in range(9, 12)),
        color="r",
        alpha=0.1,
    )
    ax.fill_between(
        df_test["date"],
        min(y_test),
        max(y_test),
        where=pd.Series.apply(df_test["month"], lambda month: month in [12, 1, 2]),
        color="c",
        alpha=0.2,
    )
    ax.scatter(df_test["date"], y_test, color="blue", label="Actual")
    ax.plot(df_test["date"], y_pred, color="red", label="Predicted")
    ax.grid()
    ax.set_xlabel("Date")
    ax.set_ylabel("Intensity")
    ax.set_title("Carbon Intensity")
    ax.legend()
    plt.savefig("/app/imgs/extra.png")


main()
