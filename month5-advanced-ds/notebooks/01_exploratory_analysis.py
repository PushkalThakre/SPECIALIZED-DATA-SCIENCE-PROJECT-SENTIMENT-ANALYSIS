"""
01_exploratory_analysis.py
EDA for both datasets. Run with: python notebooks/01_exploratory_analysis.py
Or open as Jupyter: jupyter nbconvert --to notebook --execute this file
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── Customer Churn EDA ────────────────────────────────────────
print("\n[1] Customer Churn Dataset")
df_churn = pd.read_csv("data/raw/customer_churn.csv")
print(f"  Shape: {df_churn.shape}")
print(f"  Churn Rate: {df_churn['Churn'].mean()*100:.1f}%")
print(f"  Missing values:\n{df_churn.isnull().sum()[df_churn.isnull().sum()>0]}")
print(f"\n  Numeric Stats:\n{df_churn[['Tenure','MonthlyCharges','TotalCharges']].describe().round(2)}")
print(f"\n  Contract Distribution:\n{df_churn['Contract'].value_counts()}")
print(f"\n  Churn by Contract:\n{df_churn.groupby('Contract')['Churn'].mean().round(3)}")

# Churn visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Customer Churn — EDA", fontsize=14, fontweight="bold")

df_churn["Churn"].value_counts().plot(kind="bar", ax=axes[0,0], color=["steelblue","tomato"])
axes[0,0].set_title("Churn Distribution"); axes[0,0].set_xticklabels(["No Churn","Churn"], rotation=0)

df_churn["Tenure"].hist(bins=20, ax=axes[0,1], color="steelblue", edgecolor="white")
axes[0,1].set_title("Tenure Distribution")

df_churn["MonthlyCharges"].hist(bins=20, ax=axes[0,2], color="orange", edgecolor="white")
axes[0,2].set_title("Monthly Charges Distribution")

df_churn.groupby("Contract")["Churn"].mean().plot(kind="bar", ax=axes[1,0], color="coral")
axes[1,0].set_title("Churn Rate by Contract"); axes[1,0].tick_params(axis="x", rotation=30)

df_churn.boxplot(column="MonthlyCharges", by="Churn", ax=axes[1,1])
axes[1,1].set_title("Monthly Charges by Churn")

corr_cols = ["Tenure","MonthlyCharges","TotalCharges","Churn","SeniorCitizen"]
sns.heatmap(df_churn[corr_cols].corr(), annot=True, fmt=".2f", ax=axes[1,2], cmap="coolwarm")
axes[1,2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/churn_eda.png", dpi=100, bbox_inches="tight")
print(f"\n  ✔ EDA plot saved: {OUTPUT_DIR}/churn_eda.png")

# ── Supermarket Sales EDA ─────────────────────────────────────
print("\n[2] Supermarket Sales Dataset")
df_sales = pd.read_csv("data/raw/supermarket_sales.csv", parse_dates=["Date"])
print(f"  Shape: {df_sales.shape}")
print(f"  Date range: {df_sales['Date'].min().date()} → {df_sales['Date'].max().date()}")
print(f"  Total Revenue: ₹{df_sales['Total'].sum():,.2f}")
print(f"  Top Product Line:\n{df_sales.groupby('Product_Line')['Total'].sum().sort_values(ascending=False)}")

daily = df_sales.groupby("Date")["Total"].sum().reset_index()

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Supermarket Sales — EDA", fontsize=14, fontweight="bold")

axes[0,0].plot(daily["Date"], daily["Total"], color="steelblue", linewidth=1.2)
axes[0,0].set_title("Daily Total Sales"); axes[0,0].tick_params(axis="x", rotation=30)

df_sales.groupby("Product_Line")["Total"].sum().plot(kind="barh", ax=axes[0,1], color="teal")
axes[0,1].set_title("Revenue by Product Line")

df_sales["Payment"].value_counts().plot(kind="pie", ax=axes[1,0], autopct="%1.1f%%")
axes[1,0].set_title("Payment Methods")

df_sales.groupby("Branch")["Total"].mean().plot(kind="bar", ax=axes[1,1], color=["#1f77b4","#ff7f0e","#2ca02c"])
axes[1,1].set_title("Avg Sale by Branch"); axes[1,1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sales_eda.png", dpi=100, bbox_inches="tight")
print(f"\n  ✔ EDA plot saved: {OUTPUT_DIR}/sales_eda.png")
print("\n  EDA Complete!")
