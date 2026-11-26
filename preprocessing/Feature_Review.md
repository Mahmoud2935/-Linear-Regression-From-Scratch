### Feature: HouseAge

Formula:
HouseAge = 2026 - YearBuilt

Reason:
House age is easier for humans to interpret than build year.

Result:
No measurable improvement in R², RMSE, or MSE.

Conclusion:
Linear Regression was already able to capture the information contained in YearBuilt.
HouseAge does not add new information.

### AreaPerBedroom

Rejected

Reason:
Does not represent the actual usage of property area.
Property area is not distributed only across bedrooms.
Also not suitable for offices, studios, warehouses, etc.

### Floor

**Hypothesis:**
Higher floors may increase property value.

**Result:**
No clear relationship was found between Floor and SalePrice in the scatter plot.

**Conclusion:**
Although the idea makes sense from a human perspective, the data showed that the model does not gain useful information from Floor alone.

**Decision:**
No feature engineering will be created from Floor.

### NeighborhoodScore

**Idea:**
Convert each neighborhood into a numerical score using the average SalePrice of that neighborhood.

**Reason:**
Location is one of the strongest factors affecting real estate prices, so it was expected that a neighborhood score could help the model understand location value more directly.

**Result:**

Before:
R² = 0.9528
RMSE = 259,622

After:
R² = 0.9511
RMSE = 264,182

**Conclusion:**
The feature slightly reduced performance.

The likely reason is that neighborhood information was already available through one-hot encoded neighborhood features, so the new feature did not provide meaningful additional information.

**Decision:**
Rejected.

## General Observation

Several feature engineering attempts were tested, including:

- HouseAge
- NeighborhoodScore
- Floor analysis

None of them produced a meaningful improvement in model performance.

Even when removing ListPrice, model performance dropped significantly and the engineered features were still unable to recover that loss.

### Conclusion

The current Linear Regression model may have already extracted most of the useful information it can capture from the dataset.

This suggests one of two possibilities:

1. The dataset has reached its practical information limit for a Linear Regression model.
2. Further improvement may require a different model rather than additional feature engineering.

### Next Step

Instead of continuing to create more engineered features, it may be more valuable to experiment with other machine learning models and compare their performance against Linear Regression.