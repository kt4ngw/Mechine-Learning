import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

X = np.array([[150, 200, 250, 300, 350, 400, 600]]).reshape(7, 1)
Y = np.array([[6450, 7450, 8450, 9450, 11450, 15450, 18450]]).reshape(7, 1)
# 建立线性回归模型
regr = linear_model.LinearRegression()
# 拟合
regr.fit(X, Y)
# 不难得到直线的斜率、截距
a, b = regr.coef_, regr.intercept_
# 给出预测面积，预测房子价格price
area = np.array([[238.5]]).reshape(-1, 1)

# 作图
# 1.真实数据的点
plt.scatter(X, Y, color='blue', label='原始数据点')
# 2.拟合的直线
plt.plot(X, regr.predict(X), color='red', linewidth=4, label='拟合线')
plt.xlabel("square_feet")
plt.ylabel("price")
plt.grid()
plt.legend()
plt.show()