#import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates 
import pandas as pd

dataset = pd.read_csv(r"‪finesse.csv")

date = []
open_v = []


date = dataset.iloc[1:20, 0].values

open_v = dataset.iloc[1:20, 1].values

# test = dates.datestr2num(dataset.iloc[20:30 , 0].values);

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0,max_depth=3)



converted_dates = dates.datestr2num(date)
x_axis = (converted_dates)

date = date.reshape(-1,1)
# print(date)
x_axis = x_axis.reshape(-1,1)
# test = test.reshape(-1,1)

regressor.fit(x_axis, open_v)

# Uncomment the next line to predict 
# y_pred = regressor.predict(test) 


#date_grid = np.arange(stop = 20)
#date_grid = date_grid.reshape((len(date_grid), 1))
date_grid = date.reshape((len(x_axis), 1))

plt.scatter(converted_dates, open_v, color = 'red')
plt.plot(x_axis, regressor.predict(x_axis), color = 'blue')
# plt.plot(x_axis, open_v, color = 'green')
plt.xlabel('Dates')
plt.ylabel('Opening Value')
plt.show()