""" 
Self-assessment

1. Which of the three cleaning decisions — drop, fill, or flag — feels most ambiguous to you? What additional information would you want before making that call on real data?
I think that the most ambiguous was the flag. Because I would want more information about the context and rules before falgging.
2. Look at your comments. If someone who didn't know the dataset read your script, would they understand why you made each decision? Revise any comment that only says what you did without saying why.
No, I think my first comments were made to explain it in context. I revised them to explain why I made each decision.
3. After running describe(), what is the most important thing you noticed about this dataset that you did not expect when you first loaded it?
The most important thing was the 9,999 quantity_sold placeholder. I did not expect that to be in the dataset. 

"""
# LAB

import pandas as pd

# ── EXERCISE 1: Load & Inspect ───────────────────────────────────────────────
df = pd.read_csv("data/messy_sales.csv")
print("Raw shape:", df.shape)

df.info()
print("\nMissing values:\n", df.isnull().sum())
# Data Quality Issues & Impact:
# 1. order_date and order_id have missing values, which break transaction tracking and time-series analysis.
# 2. unit_price is stored as text (object) due to literal '$' symbols and commas, preventing math.
# 3. quantity_sold contains missing values and non-numeric artifacts that distort sales volume totals.

# ── EXERCISE 2: Standardize Column Names ─────────────────────────────────────
# Convert to lowercase and replace spaces with underscores for easier, error-free column referencing.
df.columns = df.columns.str.lower().str.replace(" ", "_")
print("\nCleaned columns:", df.columns.tolist())

# ── EXERCISE 3: Handle Missing Values ────────────────────────────────────────
# Print shape before dropping essential rows
print("Shape before dropping essential rows:", df.shape)

# Drop rows missing critical identifiers because an order cannot be tracked or fulfilled without an ID and date.
df = df.dropna(subset=["order_id", "order_date"])
print("Shape after dropping essential rows:", df.shape)

# Fill missing quantity_sold with the mean to preserve sample size without heavily biasing total distribution.
df["quantity_sold"] = df["quantity_sold"].fillna(df["quantity_sold"].mean())

# ── EXERCISE 4: Fix Types & Calculate Revenue ────────────────────────────────
# Strip dollar signs and commas, then convert to float so price data can be used in arithmetic operations.
df["unit_price"] = df["unit_price"].str.replace("$", "", regex=False)
df["unit_price"] = df["unit_price"].str.replace(",", "", regex=False).astype(float)

# Convert quantity to integer for whole-item counts.
df["quantity_sold"] = df["quantity_sold"].astype(int)

# Convert order_date to datetime, forcing unparseable strings to NaT to prevent date-parsing crashes.
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
bad_dates = df["order_date"].isnull().sum()
print(f"Unparseable dates: {bad_dates}")

# Add revenue column
df["revenue"] = (df["unit_price"] * df["quantity_sold"]).round(2)
print("\nData Types:\n", df.dtypes)

# ── EXERCISE 5: Check & Remove Duplicates ────────────────────────────────────
print(f"Duplicate rows: {df.duplicated().sum()}")
# Remove exact duplicate rows to prevent double-counting sales volume and inflating revenue metrics.
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

# ── EXERCISE 6: Describe & Clean Anomalies ───────────────────────────────────
print("\nDescriptive stats after cleaning:")
print(df.describe())
# Suspicious Stats & Business Explanations:
# 1. Max quantity_sold is 9,999, which represents a test entry or typing placeholder rather than a real customer order.
# 2. Negative revenue/unit_price values represent legitimate customer returns or refunds (money flowing back to the customer).

# Drop the 9,999 placeholder quantity because it severely skews average sales volume and total revenue metrics.
df = df[df["quantity_sold"] != 9999]

# ── EXERCISE 7: Save & Reload ────────────────────────────────────────────────
# Export the cleaned dataset for downstream reporting and analytics.
df.to_csv("data/clean_sales.csv", index=False)
print("\nSaved clean_sales.csv")

print("\nReloaded without parse_dates (.info):")
print(pd.read_csv("data/clean_sales.csv").info())

print("\nReloaded with parse_dates (.info):")
df_reloaded = pd.read_csv("data/clean_sales.csv", parse_dates=["order_date"])
print(df_reloaded.info())