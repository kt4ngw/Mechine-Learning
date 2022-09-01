# 1. import required libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 2. load dataset
data_X = pd.read_csv('..\data\diabetes\X.csv', header=None, sep=' ')
data_y = pd.read_csv('..\data\diabetes\y.csv', header=None, sep=' ')
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2)

# 3. make model and predict test-data
Linear_Regr = linear_model.LinearRegression()
Linear_Regr.fit(X_train, y_train)
y_pred = Linear_Regr.predict(X_test)

# 4. evaluate model - R2 and Mse
print("Coefficients: \n", Linear_Regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

# 5. draw
fig = plt.figure(figsize=(10, 6))
plt.scatter([i for i in range(89)], y_test, color="black")
plt.plot([i for i in range(89)], y_pred, color="blue", linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()