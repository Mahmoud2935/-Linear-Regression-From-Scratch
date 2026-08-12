import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json


# Create dashboard layout

fig = plt.figure(figsize=(16, 12))

gs = fig.add_gridspec(
    4,
    2,
    height_ratios=[0.4, 1.2, 1, 1],
    hspace=0.9,
    wspace=0.30
)


# Load and display model metrics

ax0 = fig.add_subplot(gs[0, :])
ax0.axis("off")

with open(
    "results/metrics.json",
    "r"
) as file:
    metrics = json.load(file)

r2 = metrics["r2"]
mse = metrics["mse"]
rmse = metrics["rmse"]

ax0.text(
    0.5,
    0.65,
    f"R²\n{r2*100:.2f}%",
    ha="center",
    va="center",
    fontsize=14,
    bbox=dict(
        facecolor="lightgreen",
        alpha=0.8
    )
)

ax0.text(
    0.25,
    0.65,
    f"MSE\n{mse/1e9:.2f} B",
    ha="center",
    va="center",
    fontsize=14,
    bbox=dict(
        facecolor="lightblue",
        alpha=0.8
    )
)

ax0.text(
    0.75,
    0.65,
    f"RMSE\n{rmse:.4f}",
    ha="center",
    va="center",
    fontsize=14,
    bbox=dict(
        facecolor="orange",
        alpha=0.8
    )
)


# Plot training loss curve

ax1 = fig.add_subplot(gs[1, 0])

with open("results/losses.json", "r") as file:
    losses = json.load(file)

ax1.plot(
    losses,
    color="blue",
    linewidth=2
)

ax1.set_title("Loss Curve")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")


# Plot correlation heatmap

ax2 = fig.add_subplot(gs[1, 1])

correlation_matrix = pd.read_json(
    "results/correlation_matrix.json"
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax2
)

ax2.set_title("Correlation Matrix")
ax2.tick_params(axis="x", labelsize=8)
ax2.tick_params(axis="y", labelsize=8)


# Load dataset for exploratory visualizations

df = pd.read_csv(
    "data/data_cleaned_for_train.csv"
)


# Visualize the strongest feature relationship

ax3 = fig.add_subplot(gs[2, 0])

sns.scatterplot(
    data=df,
    x="ListPrice",
    y="SalePrice",
    ax=ax3
)

ax3.set_title("ListPrice vs SalePrice")
ax3.set_xlabel("List Price")
ax3.set_ylabel("Sale Price")


# Visualize a weak feature relationship

ax4 = fig.add_subplot(gs[2, 1])

sns.scatterplot(
    data=df,
    x="SqFt",
    y="SalePrice",
    ax=ax4
)

ax4.set_title("SqFt vs SalePrice")
ax4.set_xlabel("Square Feet")
ax4.set_ylabel("Sale Price")


# Compare actual values with model predictions

ax5 = fig.add_subplot(gs[3, :])

with open(
    "results/predictions.json",
    "r"
) as file:
    prediction = json.load(file)

y_true = prediction["y_true"]
y_pred = prediction["y_pred"]

sns.scatterplot(
    x=y_true,
    y=y_pred,
    ax=ax5
)

ax5.plot(
    [min(y_true), max(y_true)],
    [min(y_true), max(y_true)],
    color="red",
    linestyle="--"
)

ax5.set_title("Actual vs Predicted")
ax5.set_xlabel("Actual Price")
ax5.set_ylabel("Predicted Price")


# Final dashboard formatting and export

fig.suptitle(
    "Linear Regression From Scratch Dashboard",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

fig.subplots_adjust(
    top=0.9
)

plt.savefig(
    "final_dashboard.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()