# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Initialize two empty lists
dates = []
open_v = []

# Importing the dataset
def get_data(filename):
    
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # skipping column names
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))  # Only gets day of the month which is at index 0
            open_v.append(float(row[1]))  # Convert to float for more precision
    
            #high_v.append(float(row[2]))
            #low_v.append(float(row[3]))
            #close_v.append(float(row[4]))
            #adj_v.append(float(row[5]))
            #v.append(int(row[6]))
    return
#dataset = pd.read_csv('C:/Users/theabhishekg/Desktop/Machine Learning A-Z/Part 1 - Data Preprocessing/finesse.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
#"""from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(29)

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Random Forest Regression')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()