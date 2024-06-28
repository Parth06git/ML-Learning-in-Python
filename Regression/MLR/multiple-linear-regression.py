# Importing Labraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
df = pd.read_csv('50_Startups.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encoding the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# For manual 
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.compose import ColumnTransformer
# labelencoder_X = LabelEncoder()
# X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
# ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])], remainder='passthrough')
# X = ct.fit_transform(X)

# # Avoid Dummy variable trap for manual model building
# X = X[:, 1:]

# Spliting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(X_train, y_train)  

# predicting the test results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making a single prediction (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')

print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# Manually Applying the Backward elemination

# import statsmodels.regression.linear_model as lm
# X = np.append(arr=np.ones((50, 1)).astype(int) , values=X, axis=1)

# X_opt = np.array(X[:, [0,1,2,3,4,5]], dtype=float)
# regressor_OLS = lm.OLS(endog = y, exog=X_opt).fit() # Model is filled with all possible predictions(step-2)
# # print(regressor_OLS.summary()) # Now check step-3
# X_opt = np.array(X[:, [0,1,3,4,5]], dtype=float)
# regressor_OLS = lm.OLS(endog = y, exog=X_opt).fit() # Model is filled with all possible predictions(step-2)
# # print(regressor_OLS.summary()) # Now check step-3
# X_opt = np.array(X[:, [0,3,4,5]], dtype=float)
# regressor_OLS = lm.OLS(endog = y, exog=X_opt).fit() # Model is filled with all possible predictions(step-2)
# # print(regressor_OLS.summary()) # Now check step-3
# X_opt = np.array(X[:, [0,3,5]], dtype=float)
# regressor_OLS = lm.OLS(endog = y, exog=X_opt).fit() # Model is filled with all possible predictions(step-2)
# # print(regressor_OLS.summary()) # Now check step-3
# X_opt = np.array(X[:, [0,3]], dtype=float)
# regressor_OLS = lm.OLS(endog = y, exog=X_opt).fit() # Model is filled with all possible predictions(step-2)
# print(regressor_OLS.summary()) # Now check step-3

