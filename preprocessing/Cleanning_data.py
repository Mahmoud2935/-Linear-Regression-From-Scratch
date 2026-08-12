import pandas as pd


# Load raw housing dataset
df = pd.read_excel("data/Excel_data_salary.xlsx")


# Quick data understanding (used during exploration)

# print(df.describe())
# print('='*20)
# print(df.info())


# Explore relationships between important numerical features

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


# Remove identifier and non-useful columns

df = df.drop(
    columns=["PropertyID", "ListDate", "Agent"]
)

# print(df.columns)
# print(df.isnull().sum())


# Explore categorical columns before encoding

# print(df["PropertyType"].nunique())
# print(df["City"].nunique())
# print(df["Neighborhood"].nunique())
# print(df["Status"].nunique())

# train_df = df[
#     df["SalePrice"].notna()
# ]

# print(
#     train_df[
#         [
#             "PropertyType",
#             "City",
#             "Neighborhood",
#             "Status"
#         ]
#     ].head()
# )


# Remove Status to avoid potential data leakage

df = df.drop("Status", axis=1)


# Keep a copy before encoding for analysis and visualizations

df_before_encoding = df.copy()


# Build a clean correlation matrix for the dashboard

important_columns = [
    "SalePrice",
    "ListPrice",
    "PricePerSqFt",
    "SqFt",
    "Bedrooms",
    "Bathrooms",
    "YearBuilt"
]

corr = df_before_encoding[
    important_columns
].corr()

corr.to_json(
    "results/correlation_matrix.json"
)


# Convert categorical features into numerical features using One-Hot Encoding

df = pd.get_dummies(
    df,
    columns=[
        "PropertyType",
        "Neighborhood",
        "City"
    ],
    dtype=int
)


# Save processed dataset for model training

df.to_csv(
    "data/data_cleaned_for_train.csv",
    index=False
)