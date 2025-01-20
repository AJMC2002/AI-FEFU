import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import kagglehub

path = kagglehub.dataset_download(
    "anycaroliny/latin-america-weather-and-air-quality-data"
)
data = pd.read_csv(path + "/LA_daily_air_quality.csv")

data["date"] = pd.to_datetime(data["date"])

data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day

target = "pm10"

for i in range(1, 8):
    data[f"{target}_lag_{i}"] = data[target].shift(i)

data = data.dropna()

features = [
    "year",
    "month",
    "day",
    "latitude",
    "longitude",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
] + [f"{target}_lag_{i}" for i in range(1, 8)]

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

date_labels = pd.to_datetime(X_test[["year", "month", "day"]])
plt.figure(figsize=(10, 6))
plt.scatter(date_labels, y_test.values, label="Actual", marker="o")
plt.scatter(date_labels, y_pred, label="Predicted", marker="x")
plt.title(f"{target} Prediction for All Data")
plt.xlabel("Date")
plt.ylabel(target)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
