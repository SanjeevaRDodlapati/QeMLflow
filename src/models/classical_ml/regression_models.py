from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

class RegressionModels:
    def __init__(self, model_type='linear'):
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge()
        elif model_type == 'lasso':
            self.model = Lasso()
        else:
            raise ValueError("Unsupported model type. Choose 'linear', 'ridge', or 'lasso'.")

    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_val)
        mse = mean_squared_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
        return mse, r2

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model