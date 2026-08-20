import pandas as pd
import numpy as np
import json

from preprocessing.feature_scaling import *
from models.linear_regression import LinearRegression
from Metrics.metrices import mse, rmse, r2_score


# Load prepared dataset

df = pd.read_csv("data/data_cleaned_for_train.csv")


# Separate labeled and unlabeled samples

train_df = df[
    df["SalePrice"].notna()
]

predict_df = df[
    df["SalePrice"].isna()
]


# Split features and target

x = train_df.drop(
    ["SalePrice"],
    axis=1
)

y = train_df["SalePrice"]


# Shuffle dataset before splitting

indices = np.arange(len(x))

np.random.seed(42)

np.random.shuffle(indices)

x = x.iloc[indices]
y = y.iloc[indices]


# Train / Test Split

split_index = int(
    len(x) * 0.8
)

x_train = x[:split_index]
x_test = x[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]


# Apply feature scaling

x_train_standrize, mean, std = standardize_train(
    x_train
)


x_test_standrize = standardize_test(
    x_test,
    mean,
    std
)



# Initialize and train the model

model = LinearRegression(
    learning_rate=0.01,
    epochs=1000
)

model.fit(x_train_standrize,y_train)

# Evaluate model on unseen test data

predictions = model.predict(
    x_test_standrize
)

# save data for feature using
metadata = {
    "feature_names": x_train.columns.tolist(), # feature name because we need order later
    "mean": mean.to_dict(),  # because it is pandas series 
    "std": std.to_dict()
}

with open("save_model/metadata.json",'w')as file :
    json.dump(metadata, file, indent=4)

# save model 
model.save_model()

# load model 

loaded_model = LinearRegression.load_model()

pred_after = loaded_model.predict(
    x_test_standrize
)

# Test model after save and load 

print("_"*30)
print(predictions[:5])
print(pred_after[:5])
print("_"*30)
# train the model again to see if incremental Training work or not 
loaded_model.epochs = 50
print("Before:", loaded_model.weights[0])

loaded_model.fit(
    x_train_standrize,
    y_train
)

print("After:", loaded_model.weights[0])
# Save loss history for visualizations

with open("results/losses.json","w") as file:
    json.dump(model.losses,file,indent=4)


# Save actual and predicted values for dashboard

predictions_data = {
    "y_true": y_test.tolist(),
    "y_pred": predictions.tolist()
}

with open(
    "results/predictions.json","w") as file:
    json.dump(predictions_data,file,indent=4)

# Calculate evaluation metrics
print('='*30)
mse1 = mse(y_test,predictions)
print(f"mse: {mse1}")

rmse1 = rmse( y_test, predictions)
print(f"rmse: {rmse1}")

r2_score1 = r2_score(y_test,predictions)
print(f"R^2 Score: {r2_score1}")

# Save metrics for dashboard cards

metrics = {
    "mse": float(mse1),
    "rmse": float(rmse1),
    "r2": float(r2_score1)
}

with open(
    "results/metrics.json","w") as file:
    json.dump(metrics,file,indent=4
    )


# Display sample predictions

for true, pred in zip(y_test[:10],predictions[:10]):
    print( f"True: {true} | Pred: {pred}")


