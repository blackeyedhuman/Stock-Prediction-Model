import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import pandas as pd

dataset = pd.read_csv('C:/Users/theabhishekg/Desktop/ML FIN/YF.csv')
#dataset['Date']=pd.to_datetime(dataset['Date'], format="%Y/%m/%d")
X = []
y = []

for row in dataset:
    for column in dataset:
        if column == 0:
            X += dataset[row][0]


#X = str(dataset.iloc[1:20, 0].values)

x = dates.datestr2num(X)
#y = dataset.iloc[1:20, 1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100,min_samples_leaf=2, random_state = 0)
regressor.fit(x, y)
y_pred = regressor.predict(20)
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot_date(X_grid, regressor.predict(X_grid), color = 'blue')
plt.xlabel('Customers')
plt.ylabel('Probability')
plt.show()