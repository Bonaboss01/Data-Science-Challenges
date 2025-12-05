# ===========================================
# FULL REGRESSION MODELLING WORKFLOW (Python)
# Demonstrates: EDA, preparation, modelling, diagnostics, interpretation
# ===========================================

# --- 1. Import Required Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------------------------
# 2. Load Data
# -------------------------------------------
# For demonstration, let's create a synthetic dataset.
# Replace this with: df = pd.read_csv("your_dataset.csv")
np.random.seed(42)

df = pd.DataFrame({
    "YearsExperience": np.random.randint(1, 15, 100),
    "EducationLevel": np.random.choice([1, 2, 3], 100),  # 1=BSc, 2=MSc, 3=PhD
    "Age": np.random.randint(22, 55, 100),
    "Salary": np.random.randint(25000, 85000, 100)
})

# Preview the data
print(df.head())

# -------------------------------------------
# 3. Exploratory Data Analysis (EDA)
# -------------------------------------------
print(df.describe())  # Summary statistics
print(df.isna().sum())  # Missing values

# Quick pairplot to understand relationships
sns.pairplot(df)
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# -------------------------------------------
# 4. Define Features (X) and Target (y)
# -------------------------------------------
X = df[["YearsExperience", "EducationLevel", "Age"]]
y = df["Salary"]

# -------------------------------------------
# 5. Train–Test Split
# -------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------
# 6. Fit Baseline Linear Regression Model
# -------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

# Model coefficients
print("Intercept:", lr.intercept_)
print("Coefficients:", list(zip(X.columns, lr.coef_)))

# -------------------------------------------
# 7. Evaluate Model
# -------------------------------------------
y_pred = lr.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# -------------------------------------------
# 8. Build Statsmodels Version for Interpretability
# -------------------------------------------
# Add constant for intercept
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()

print(model_sm.summary())  # Shows p-values, confidence intervals, etc.

# -------------------------------------------
# 9. Multicollinearity Check (VIF)
# -------------------------------------------
X_vif = X_train_sm.drop(columns=[])  # Keep const included

vif_df = pd.DataFrame()
vif_df["Feature"] = X_vif.columns
vif_df["VIF"] = [
    variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])
]

print(vif_df)

# -------------------------------------------
# 10. Residual Diagnostics
# -------------------------------------------
residuals = y_train - model_sm.predict(X_train_sm)

# Residual plot
plt.scatter(model_sm.predict(X_train_sm), residuals)
plt.axhline(0, color='red')
plt.title("Residuals vs Fitted")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()

# Histogram of residuals
sns.histplot(residuals, kde=True)
plt.title("Distribution of Residuals")
plt.show()

# Q-Q plot for normality check
sm.qqplot(residuals, line='45')
plt.title("Q-Q Plot of Residuals")
plt.show()

# -------------------------------------------
# 11. Interpretation Example
# -------------------------------------------
print("\n--- INTERPRETATION EXAMPLE ---")

print("""
1️⃣ YearsExperience coefficient:
   For each additional year of experience, salary increases by approximately
   {:.2f} units, holding other variables constant.

2️⃣ EducationLevel coefficient:
   A 1-unit increase in education level (e.g., MSc → PhD) increases salary
   by {:.2f}, holding other factors constant.

3️⃣ Age coefficient:
   Each additional year of age increases salary by {:.2f}, controlling for other variables.
""".format(
    lr.coef_[0], lr.coef_[1], lr.coef_[2]
))

print("""
4️⃣ RMSE:
   Shows the average prediction error. Lower = better.

5️⃣ R²:
   Indicates percentage of variation in Salary explained by the model.

6️⃣ VIF:
   VIF > 5 suggests multicollinearity issues.

7️⃣ Residual Plots:
   Should show no pattern (random scatter) → indicates linear model is appropriate.
""")

