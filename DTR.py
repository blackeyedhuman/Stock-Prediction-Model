#import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates 
import pandas as pd

dataset = pd.read_csv('C:/Users/theabhishekg/Desktop/ML FIN/YF.csv')

date = []
open_v = []


date = dataset.iloc[1:20, 0].values

open_v = dataset.iloc[1:20, 1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0,max_depth=3)

converted_dates = dates.datestr2num(date)
x_axis = (converted_dates)

regressor.fit(x_axis, open_v)
y_pred = regressor.predict()


#date_grid = np.arange(stop = 20)
#date_grid = date_grid.reshape((len(date_grid), 1))
date_grid = date.reshape((len(x_axis), 1))

plt.scatter(date, open_v, color = 'red')
plt.plot(x_axis, regressor.predict(x_axis), color = 'blue')
plt.xlabel('Dates')
plt.ylabel('Opening Value')
plt.show()