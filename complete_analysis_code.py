# Weather and Daily Step Count Analysis - Combined Code

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

STEP_FILE = Path("apple-steps-since-2026-01-01.csv")
WEATHER_FILE = Path("open-meteo-daily-averages.csv")


def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Make sure it is in the same folder as this script.")
    return pd.read_csv(path, sep=None, engine="python")


# 1. Load data
steps_raw = read_csv_safely(STEP_FILE)
weather_raw = read_csv_safely(WEATHER_FILE)

print("Steps data shape:", steps_raw.shape)
print("Weather data shape:", weather_raw.shape)

# 2. Clean and merge data
steps_df = steps_raw.rename(columns={"total_steps": "steps"}).copy()
weather_df = weather_raw.rename(columns={
    "avg_temperature_2m_c": "temperature_c",
    "avg_precipitation_mm": "precipitation_mm"
}).copy()

required_steps_cols = {"date", "steps"}
required_weather_cols = {"date", "temperature_c", "precipitation_mm", "dominant_weather_code"}

missing_steps = required_steps_cols - set(steps_df.columns)
missing_weather = required_weather_cols - set(weather_df.columns)

if missing_steps:
    raise ValueError(f"Missing required step columns: {missing_steps}")
if missing_weather:
    raise ValueError(f"Missing required weather columns: {missing_weather}")

steps_df["date"] = pd.to_datetime(steps_df["date"], errors="coerce")
weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce")

steps_df["steps"] = pd.to_numeric(steps_df["steps"], errors="coerce")
weather_df["temperature_c"] = pd.to_numeric(weather_df["temperature_c"], errors="coerce")
weather_df["precipitation_mm"] = pd.to_numeric(weather_df["precipitation_mm"], errors="coerce")
weather_df["dominant_weather_code"] = pd.to_numeric(weather_df["dominant_weather_code"], errors="coerce")

steps_df = steps_df.dropna(subset=["date", "steps"])
weather_df = weather_df.dropna(subset=["date", "temperature_c", "precipitation_mm", "dominant_weather_code"])

df = pd.merge(steps_df, weather_df, on="date", how="inner")
df = df.sort_values("date").reset_index(drop=True)

df["rainy"] = df["precipitation_mm"] > 0
df["weekday"] = df["date"].dt.day_name()
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["month"] = df["date"].dt.month

print("Merged data shape:", df.shape)
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
print("Missing values after cleaning:")
print(df.isna().sum())

# 3. Descriptive statistics
summary_cols = ["steps", "temperature_c", "precipitation_mm", "dominant_weather_code", "is_weekend", "month"]
print(df[summary_cols].describe().T)

rainy_steps = df.loc[df["rainy"], "steps"]
non_rainy_steps = df.loc[~df["rainy"], "steps"]

print("Number of rainy days:", int(df["rainy"].sum()))
print("Number of non-rainy days:", int((~df["rainy"]).sum()))
print("Mean steps on non-rainy days:", round(non_rainy_steps.mean(), 2))
print("Mean steps on rainy days:", round(rainy_steps.mean(), 2))

# 4. EDA graphs
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["steps"], marker="o")
plt.title("Daily Steps Over Time")
plt.xlabel("Date")
plt.ylabel("Steps")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["temperature_c"], df["steps"], alpha=0.8)
plt.title("Temperature vs Daily Steps")
plt.xlabel("Average Temperature (C)")
plt.ylabel("Steps")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["precipitation_mm"], df["steps"], alpha=0.8)
plt.title("Precipitation vs Daily Steps")
plt.xlabel("Average Precipitation (mm)")
plt.ylabel("Steps")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.boxplot([non_rainy_steps, rainy_steps], labels=["Non-rainy", "Rainy"])
plt.title("Step Counts on Rainy vs Non-rainy Days")
plt.ylabel("Steps")
plt.tight_layout()
plt.show()

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_steps = df.groupby("weekday")["steps"].mean().reindex(weekday_order)

plt.figure(figsize=(9, 5))
plt.bar(weekday_steps.index, weekday_steps.values)
plt.title("Average Steps by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Average Steps")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

corr_cols = ["steps", "temperature_c", "precipitation_mm", "dominant_weather_code", "is_weekend", "month"]
corr = df[corr_cols].corr(numeric_only=True)
print(corr)

# 5. Hypothesis testing
# H0: Average steps are the same on rainy and non-rainy days.
t_stat, p_value_t = ttest_ind(rainy_steps, non_rainy_steps, equal_var=False)
u_stat, p_value_u = mannwhitneyu(rainy_steps, non_rainy_steps, alternative="two-sided")

print("Welch t-test")
print("T-statistic:", round(t_stat, 4))
print("P-value:", round(p_value_t, 4))
print("Mann-Whitney U test")
print("U-statistic:", round(u_stat, 4))
print("P-value:", round(p_value_u, 4))

for feature in ["temperature_c", "precipitation_mm"]:
    pearson_corr, pearson_p = pearsonr(df[feature], df["steps"])
    spearman_corr, spearman_p = spearmanr(df[feature], df["steps"])
    print(f"Feature: {feature}")
    print(f"  Pearson r = {pearson_corr:.4f}, p-value = {pearson_p:.4f}")
    print(f"  Spearman rho = {spearman_corr:.4f}, p-value = {spearman_p:.4f}")

# 6. Machine learning
feature_cols = [
    "temperature_c",
    "precipitation_mm",
    "dominant_weather_code",
    "rainy",
    "is_weekend",
    "day_of_week",
    "month",
]
target_col = "steps"

model_df = df.dropna(subset=feature_cols + [target_col]).copy()
model_df["rainy"] = model_df["rainy"].astype(int)

X = model_df[feature_cols]
y = model_df[target_col]

split_index = int(len(model_df) * 0.75)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

linear_preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), feature_cols)
    ]
)

tree_preprocessor = ColumnTransformer(
    transformers=[("num", SimpleImputer(strategy="median"), feature_cols)]
)

models = {
    "Baseline Mean": Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyRegressor(strategy="mean"))
    ]),
    "Linear Regression": Pipeline(steps=[
        ("preprocess", linear_preprocessor),
        ("model", LinearRegression())
    ]),
    "Ridge Regression": Pipeline(steps=[
        ("preprocess", linear_preprocessor),
        ("model", Ridge(alpha=1.0))
    ]),
    "Decision Tree": Pipeline(steps=[
        ("preprocess", tree_preprocessor),
        ("model", DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42))
    ]),
    "KNN Regression": Pipeline(steps=[
        ("preprocess", linear_preprocessor),
        ("model", KNeighborsRegressor(n_neighbors=5))
    ]),
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    predictions[name] = preds
    results.append({
        "model": name,
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds),
    })

results_df = pd.DataFrame(results).sort_values("MAE")
print(results_df)

best_model = results_df.iloc[0]["model"]
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions[best_model], alpha=0.8)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--")
plt.title(f"Actual vs Predicted Steps - {best_model}")
plt.xlabel("Actual Steps")
plt.ylabel("Predicted Steps")
plt.tight_layout()
plt.show()

# 7. Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2",
        },
        return_train_score=False,
    )
    cv_results.append({
        "model": name,
        "CV_MAE_mean": -scores["test_MAE"].mean(),
        "CV_MAE_std": scores["test_MAE"].std(),
        "CV_RMSE_mean": -scores["test_RMSE"].mean(),
        "CV_R2_mean": scores["test_R2"].mean(),
    })

cv_results_df = pd.DataFrame(cv_results).sort_values("CV_MAE_mean")
print(cv_results_df)

# 8. Decision Tree feature importance
tree_model = models["Decision Tree"]
tree_model.fit(X_train, y_train)

importances = tree_model.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances,
}).sort_values("importance", ascending=False)
print(importance_df)

plt.figure(figsize=(8, 5))
plt.bar(importance_df["feature"], importance_df["importance"])
plt.title("Decision Tree Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 9. Save cleaned dataset
df.to_csv("cleaned_weather_steps_merged.csv", index=False)
print("Saved cleaned merged dataset to cleaned_weather_steps_merged.csv")
