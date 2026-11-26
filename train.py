import json
import numpy as np
import pandas as pd

from preprocessing.feature_scaling import *
from preprocessing.Feature_Engineering import create_features

from models.linear_regression import LinearRegression
from Metrics.metrices import mse, rmse, r2_score


# ==================================================
# 1. Load Dataset
# ==================================================

df = pd.read_csv(
    "data/data_cleaned_for_train.csv"
)


# ==================================================
# 2. Separate Labeled Samples
# ==================================================

train_df = df[
    df["SalePrice"].notna()
]

predict_df = df[
    df["SalePrice"].isna()
]


# ==================================================
# 3. Apply Feature Engineering
# ==================================================

# train_df = create_features(
#     train_df
# )


# ==================================================
# 4. Split Features & Target
# ==================================================

x = train_df.drop(
    "SalePrice",
    axis=1
)

y = train_df["SalePrice"]

print(train_df.shape)
# ==================================================
# 5. Shuffle Dataset
# ==================================================

np.random.seed(42)

indices = np.arange(len(x))
np.random.shuffle(indices)

x = x.iloc[indices]
y = y.iloc[indices]


# ==================================================
# 6. Train / Test Split
# ==================================================

split_index = int(
    len(x) * 0.8
)

x_train = x[:split_index]
x_test = x[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]


# ==================================================
# 7. Feature Scaling
# ==================================================

x_train_standardized, mean, std = (
    standardize_train(
        x_train
    )
)

x_test_standardized = (
    standardize_test(
        x_test,
        mean,
        std
    )
)


# ==================================================
# 8. Train Linear Regression
# ==================================================

model = LinearRegression(
    learning_rate=0.01,
    epochs=1000
)

model.fit(
    x_train_standardized,
    y_train
)

# ==================================================
# 8. Predict On Training Data
# ==================================================

train_predictions = model.predict(
    x_train_standardized
)

# ==================================================
# 9. Make Predictions
# ==================================================

predictions = model.predict(
    x_test_standardized
)



# ==================================================
# 10. Evaluate Model
# ==================================================

mse_score = mse(
    y_test,
    predictions
)

rmse_score = rmse(
    y_test,
    predictions
)

r2_test = r2_score(
    y_test,
    predictions
)

train_r2 = r2_score(
    y_train,
    train_predictions
)

print("=" * 30)
print(f"MSE  : {mse_score}")
print(f"RMSE : {rmse_score}")
print(f"R²   : {r2_test}")
print(f"R^2  : {train_r2}")



# ==================================================
# 11. Save Metadata
# ==================================================

metadata = {
    "feature_names": x_train.columns.tolist(),
    "mean": mean.to_dict(),
    "std": std.to_dict()
}

with open(
    "save_model/metadata.json",
    "w"
) as file:
    json.dump(
        metadata,
        file,
        indent=4
    )


# ==================================================
# 12. Save Model
# ==================================================

model.save_model()


# ==================================================
# 13. Verify Save / Load
# ==================================================

loaded_model = (
    LinearRegression.load_model()
)

loaded_predictions = (
    loaded_model.predict(
        x_test_standardized
    )
)

print("_" * 30)
print(predictions[:5])
print(loaded_predictions[:5])
print("_" * 30)


# ==================================================
# 14. Test Incremental Learning
# ==================================================

loaded_model.epochs = 50

print(
    "Before:",
    loaded_model.weights[0]
)

loaded_model.fit(
    x_train_standardized,
    y_train
)

print(
    "After:",
    loaded_model.weights[0]
)


# ==================================================
# 15. Save Dashboard Files
# ==================================================

pred_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": predictions
})

pred_df.to_csv(
    "results/predictions.csv",
    index=False
)


loss_df = pd.DataFrame({
    "Epoch": range(
        1,
        len(model.losses) + 1
    ),
    "Loss": model.losses
})

loss_df.to_csv(
    "results/losses.csv",
    index=False
)

metrics_df = pd.DataFrame(
    [{
        "mse": mse_score,
        "rmse": rmse_score,
        "r2_train": train_r2,
        "r2_test": r2_test
    }]
)

metrics_df.to_csv(
    "results/metrics.csv",
    index=False
)


# ==================================================
# 16. Display Sample Predictions
# ==================================================

for true, pred in zip(
    y_test[:10],
    predictions[:10]
):
    print(
        f"True: {true} | Pred: {pred}"
    )

#==================================================
#17. 
#==================================================

def predict_new_samples(new_samples, target_col_name="Target"):
    # Load Metadata
    with open("save_model/metadata.json", "r") as file:
        metadata = json.load(file)
        mean = pd.Series(metadata["mean"])
        std = pd.Series(metadata["std"])
    
    # Select & reorder columns to match exact training features (mean.index holds feature names in training order)
    features = new_samples[mean.index]
    scaled_samples = (features - mean) / std
    
    # Load model & predict into a separate column
    loaded_model = LinearRegression.load_model("save_model/model.json")
    result_df = new_samples.copy()
    result_df[f"Predicted_{target_col_name}"] = loaded_model.predict(scaled_samples)
    
    return result_df

predicted_df = predict_new_samples(predict_df,target_col_name= "SalePrice")

# 3. عرض أول 10 صفوف عشان تتأكد إن القيمة الأصلية NaN والجديدة اتحسبت + تاخد Screenshot
print(f"\n--- Imputed {len(predicted_df)} Missing Rows ---")
print(predicted_df[["SalePrice", "Predicted_SalePrice"]].head(10))


predicted_df.to_csv("data/predicted_SalePrice.csv", index=False)

