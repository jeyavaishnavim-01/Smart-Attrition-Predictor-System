"""
Generates a synthetic HR dataset.

"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED        = 42
N_RECORDS   = 1500
OUTPUT_PATH = Path(__file__).parent / "hr_data.csv"

np.random.seed(SEED)

DEPARTMENTS     = ["Sales", "Research & Development", "Human Resources"]
JOB_ROLES       = ["Sales Executive", "Research Scientist", "Laboratory Technician",
                    "Manufacturing Director", "Healthcare Representative",
                    "Manager", "Sales Representative", "Research Director"]
BUSINESS_TRAVEL = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
EDUCATION_FIELD = ["Life Sciences", "Other", "Medical",
                    "Marketing", "Technical Degree", "Human Resources"]
MARITAL_STATUS  = ["Single", "Married", "Divorced"]

def generate():
    df = pd.DataFrame({
        "Age":                     np.random.randint(18, 60, N_RECORDS),
        "BusinessTravel":          np.random.choice(BUSINESS_TRAVEL, N_RECORDS, p=[0.30, 0.50, 0.20]),
        "DailyRate":               np.random.randint(100, 1500, N_RECORDS),
        "Department":              np.random.choice(DEPARTMENTS, N_RECORDS),
        "DistanceFromHome":        np.random.randint(1, 30, N_RECORDS),
        "Education":               np.random.randint(1, 5, N_RECORDS),
        "EducationField":          np.random.choice(EDUCATION_FIELD, N_RECORDS),
        "EnvironmentSatisfaction": np.random.randint(1, 5, N_RECORDS),
        "Gender":                  np.random.choice(["Male", "Female"], N_RECORDS),
        "HourlyRate":              np.random.randint(30, 100, N_RECORDS),
        "JobInvolvement":          np.random.randint(1, 5, N_RECORDS),
        "JobLevel":                np.random.randint(1, 5, N_RECORDS),
        "JobRole":                 np.random.choice(JOB_ROLES, N_RECORDS),
        "JobSatisfaction":         np.random.randint(1, 5, N_RECORDS),
        "MaritalStatus":           np.random.choice(MARITAL_STATUS, N_RECORDS, p=[0.30, 0.50, 0.20]),
        "MonthlyIncome":           np.random.randint(1000, 20000, N_RECORDS),
        "MonthlyRate":             np.random.randint(2000, 27000, N_RECORDS),
        "NumCompaniesWorked":      np.random.randint(0, 10, N_RECORDS),
        "OverTime":                np.random.choice(["Yes", "No"], N_RECORDS, p=[0.30, 0.70]),
        "PercentSalaryHike":       np.random.randint(10, 26, N_RECORDS),
        "PerformanceRating":       np.random.randint(1, 5, N_RECORDS),
        "RelationshipSatisfaction":np.random.randint(1, 5, N_RECORDS),
        "StockOptionLevel":        np.random.randint(0, 4, N_RECORDS),
        "TotalWorkingYears":       np.random.randint(0, 40, N_RECORDS),
        "TrainingTimesLastYear":   np.random.randint(0, 7, N_RECORDS),
        "WorkLifeBalance":         np.random.randint(1, 5, N_RECORDS),
        "YearsAtCompany":          np.random.randint(0, 40, N_RECORDS),
        "YearsInCurrentRole":      np.random.randint(0, 18, N_RECORDS),
        "YearsSinceLastPromotion": np.random.randint(0, 15, N_RECORDS),
        "YearsWithCurrManager":    np.random.randint(0, 17, N_RECORDS),
    })

    score = (
        (df["OverTime"]         == "Yes"             ).astype(float) * 0.25 +
        (df["JobSatisfaction"]  <= 2                  ).astype(float) * 0.20 +
        (df["WorkLifeBalance"]  <= 2                  ).astype(float) * 0.15 +
        (df["DistanceFromHome"] >  20                 ).astype(float) * 0.10 +
        (df["YearsAtCompany"]   <  2                  ).astype(float) * 0.10 +
        (df["MonthlyIncome"]    <  3000               ).astype(float) * 0.10 +
        (df["MaritalStatus"]    == "Single"           ).astype(float) * 0.05 +
        (df["BusinessTravel"]   == "Travel_Frequently").astype(float) * 0.05 +
        np.random.uniform(0, 0.10, N_RECORDS)
    )

    df["Attrition"] = (score > 0.45).map({True: "Yes", False: "No"})
    df.to_csv(OUTPUT_PATH, index=False)

    rate = (df["Attrition"] == "Yes").mean()
    print(f"Generated {N_RECORDS:,} records  |  Attrition rate: {rate:.1%}")
    print(f"Saved → {OUTPUT_PATH}")

if __name__ == "__main__":
    generate()
