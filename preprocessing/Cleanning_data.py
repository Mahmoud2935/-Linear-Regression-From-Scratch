import pandas as pd


# ==================================================
# 1. Load Raw Dataset
# ==================================================

df = pd.read_excel(
    "data/Excel_data_salary.xlsx"
)


# ==================================================
# 2. Initial Exploration
# ==================================================

# Used during EDA phase

# print(df.describe())
# print(df.info())

# Numerical relationships

corr_matrix = df[
    [
        "ListPrice",
        "SalePrice",
        "PricePerSqFt",
        "SqFt",
        "Bedrooms",
        "Bathrooms"
    ]
].corr()

print(corr_matrix)


# ==================================================
# 3. Remove Non-Useful Columns
# ==================================================

df = df.drop(
    columns=[
        "PropertyID",
        "ListDate",
        "Agent"
    ]
)


# ==================================================
# 4. Feature Review Decisions
# ==================================================

# Status was removed because it may
# contain information related to
# the sale outcome.

df = df.drop(
    "Status",
    axis=1
)

# DaysOnMarket was removed because
# it is not usually available at
# prediction time.

df = df.drop(
    columns=[
        "DaysOnMarket"
    ]
)


# ==================================================
# 5. Preserve Dataset Before Encoding
# ==================================================

# Used for analysis and dashboard generation

df_before_encoding = (
    df.copy()
)


# ==================================================
# 6. Generate Correlation Matrix
# ==================================================

important_columns = [
    "SalePrice",
    "ListPrice",
    "PricePerSqFt",
    "SqFt",
    "Bedrooms",
    "Bathrooms",
    "YearBuilt"
]

corr = (
    df_before_encoding[
        important_columns
    ].corr()
)

corr.to_json(
    "results/correlation_matrix.json"
)


# ==================================================
# 7. Encode Categorical Features
# ==================================================

df = pd.get_dummies(
    df,
    columns=[
        "PropertyType",
        "Neighborhood",
        "City"
    ],
    dtype=int
)


# ==================================================
# 8. Save Processed Dataset
# ==================================================

df.to_csv(
    "data/data_cleaned_for_train.csv",
    index=False
)

# ===================================================
#9. save data_cleaned_befor_Encoding
#====================================================

df_before_encoding.to_csv("data/data_cleaned_befor_Encoding.csv")