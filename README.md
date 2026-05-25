 -DSA210-DataScienceProject
DSA210 Introduction to Data Science term project
 DSA210 Data Science Project

 Weather and Daily Step Count Analysis

Project Description

This project investigates the relationship between daily weather conditions and daily walking activity.  
The main goal is to understand whether weather variables can explain or predict daily step count.

The analysis combines two datasets:

- Apple Health daily step count data
- Open-Meteo daily weather data

The project includes data cleaning, exploratory data analysis (EDA), visualization, hypothesis testing, and machine learning applications.

Research Questions

- Is daily step count related to average temperature?
- Is daily step count related to precipitation?
- Are rainy days and non-rainy days different in terms of daily steps?
- Can weather and calendar variables help predict daily step count?

Data Sources

- **Apple Health** — daily total step count
- **Open-Meteo** — daily average temperature, precipitation, and weather code

Data Preparation

The datasets were cleaned and merged using the date column.

The preparation process included:

- converting date columns to datetime format
- converting numerical columns to numeric values
- removing invalid or missing rows
- merging the step-count dataset and weather dataset by date
- creating new variables such as rainy, weekday, day_of_week, is_weekend, and month

After cleaning and merging, the final dataset included **102 daily observations** from **2026-01-01 to 2026-04-12**.

Dataset Overview

| Dataset | Size | Notes |
|---|---:|---|
| Apple Health steps | 104 rows | Daily total step count |
| Open-Meteo weather | 102 rows | Daily temperature and precipitation |
| Merged dataset | 102 rows | Final dataset used for analysis |

Main variables used in the analysis:

- `steps`
- `temperature_c`
- `precipitation_mm`
- `dominant_weather_code`
- `rainy`
- `weekday`
- `day_of_week`
- `is_weekend`
- `month`

 Exploratory Data Analysis (EDA)

EDA was conducted to understand the general patterns in the dataset.

The analysis focused on:

- daily step count changes over time
- relationship between temperature and steps
- relationship between precipitation and steps
- differences between rainy and non-rainy days
- average steps by weekday
- correlation between variables

The average daily step count was about **5,116 steps**.  
There were **79 rainy days** and **23 non-rainy days** in the final dataset.

Average steps by weather type:

| Weather Type | Average Steps |
|---|---:|
| Non-rainy days | 4,893.78 |
| Rainy days | 5,180.61 |

Rainy days had slightly higher average steps, but the difference was small.

Average steps by weekday:

| Weekday | Average Steps |
|---|---:|
| Monday | 5,066 |
| Tuesday | 6,221 |
| Wednesday | 5,278 |
| Thursday | 5,035 |
| Friday | 5,140 |
| Saturday | 4,336 |
| Sunday | 4,816 |

Tuesday had the highest average step count, while Saturday had the lowest.

 Correlation Analysis

The correlation analysis showed weak relationships between step count and the weather variables.

| Variable | Correlation with Steps |
|---|---:|
| month | -0.3248 |
| is_weekend | -0.1529 |
| precipitation_mm | -0.1166 |
| temperature_c | -0.0424 |
| dominant_weather_code | 0.0423 |

Temperature had almost no relationship with steps.  
Precipitation also showed a weak relationship.  
The strongest correlation was with month, which suggests that time or routine may matter more than weather.

Hypothesis Testing

 Hypothesis 1: Rainy vs. Non-rainy Days

The first hypothesis test examined whether rainy and non-rainy days had different average step counts.

- **H0:** Average step count is the same on rainy and non-rainy days.
- **H1:** Average step count is different on rainy and non-rainy days.

| Test | Statistic | P-value | Result |
|---|---:|---:|---|
| Welch t-test | 0.6272 | 0.5334 | Not significant |
| Mann-Whitney U test | 972.0000 | 0.6139 | Not significant |

Since both p-values are greater than 0.05, I failed to reject the null hypothesis.  
This means there is not enough evidence to say that rainy and non-rainy days have different step counts.

 Hypothesis 2: Weather Variables and Steps

Pearson and Spearman correlation tests were used to test the relationship between weather variables and daily steps.

| Feature | Pearson r | Pearson p-value | Spearman rho | Spearman p-value |
|---|---:|---:|---:|---:|
| Temperature | -0.0424 | 0.6725 | 0.0234 | 0.8154 |
| Precipitation | -0.1166 | 0.2434 | -0.0042 | 0.9664 |

All p-values are greater than 0.05.  
Therefore, the results are not statistically significant.

Machine Learning

The machine learning task is a regression problem because the target variable is daily step count.

The following models were used:

- Baseline Mean Model
- Linear Regression
- Ridge Regression
- Decision Tree Regression
- KNN Regression

The dataset was split chronologically:

- first 75% of the data for training
- last 25% of the data for testing

The test period was from **2026-03-18 to 2026-04-12**.

 Model Results

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Decision Tree | 1,833.29 | 2,271.58 | -0.191 |
| Baseline Mean | 1,887.56 | 2,305.66 | -0.227 |
| KNN Regression | 2,023.72 | 2,354.07 | -0.279 |
| Ridge Regression | 2,125.81 | 2,446.70 | -0.381 |
| Linear Regression | 2,147.39 | 2,469.25 | -0.407 |

The Decision Tree model had the lowest test MAE.  
However, all R2 values were negative, which means the models were not strong predictors of daily step count.

Cross-validation results also showed limited performance.

| Model | CV MAE Mean | CV RMSE Mean | CV R2 Mean |
|---|---:|---:|---:|
| Baseline Mean | 1,798.10 | 2,232.22 | -0.006 |
| Ridge Regression | 1,824.99 | 2,267.89 | -0.082 |
| Linear Regression | 1,828.43 | 2,271.51 | -0.086 |
| KNN Regression | 1,895.92 | 2,341.79 | -0.129 |
| Decision Tree | 2,081.50 | 2,636.00 | -0.466 |

 Main Findings

- Weather variables did not strongly explain daily step count.
- Rainy days had slightly higher average steps than non-rainy days, but the difference was not statistically significant.
- Temperature and precipitation had weak and statistically insignificant relationships with steps.
- Machine learning models had limited predictive performance.
- Calendar and personal routine variables may be more important than weather alone.

 Time Period

2026-01-01 to 2026-04-12

Files

| File | Description |
|---|---|
| `README.md` | Final project report |
| `analysis.ipynb` | Main Jupyter notebook containing the full analysis |
| `complete_analysis_code.py` | Python script version of the analysis |
| `apple-steps-since-2026-01-01.csv` | Apple Health step-count data |
| `open-meteo-daily-averages.csv` | Weather data |
| `cleaned_weather_steps_merged.csv` | Cleaned and merged dataset |
| `requirements.txt` | Required Python libraries |
| `report_assets/` | Graphs used in the report |

 Technologies Used

- Python
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn

 Limitations

- The dataset is small and only includes 102 daily observations.
- The weather data uses daily averages, which may hide hourly effects.
- Personal factors such as exams, class schedule, sleep, transportation, and location are not included.
- Correlation does not prove causation.
- More months of data would make the analysis stronger.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Jupyter notebook:

```bash
jupyter notebook analysis.ipynb
```

Or run the Python script:

```bash
python complete_analysis_code.py
```

## Conclusion

Overall, this project did not find strong evidence that daily weather conditions explain or predict daily step count.  
The results suggest that personal routine and calendar-related factors may have a stronger effect on walking activity than weather variables alone.
