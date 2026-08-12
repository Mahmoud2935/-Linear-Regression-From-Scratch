import  numpy as np

# Mean Squared Error
def mse(y_true, y_pred):
     Error = y_true - y_pred        
     cost = (Error)**2 
     return sum(cost) / len(y_true)

# Root Mean Squared Error
def rmse(y_true,y_pred):
    return np.sqrt(mse(y_true,y_pred))

# R² Score
def r2_score(y_true,y_pred):
     total_variance = np.sum((y_true - np.mean(y_true))**2)
     remining_error = np.sum((y_true - y_pred)**2 )

     return 1- (remining_error/total_variance)  