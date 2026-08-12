# Standardize training data and return mean and std
def standardize_train(x_train):

    mean = x_train.mean()
    std = x_train.std()

    # Avoid division by zero for constant features
    std = std.replace(0, 1)

    x_train_std = (x_train - mean) / std

    return x_train_std, mean, std


# Standardize test data using training statistics
# to prevent data leakage
def standardize_test(x_test, mean, std):

    x_test_std = (x_test - mean) / std

    return x_test_std