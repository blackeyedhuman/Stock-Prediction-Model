import csv
import pandas
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Initialize two empty lists
dates = []
open_v = []
#high_v = []
#low_v = []
#adj_v = []
#v = []

def get_data(filename):
        #next(csvFileReader)  # skipping column names
        colnames = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        csvFileReader = pandas.read_csv(filename, skiprows = [1], names=colnames)
        csvFileReader.readline()
        #dates = csvFileReader.Date.tolist();
        #open_v = csvFileReader.Open.tolist();
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))  # Only gets day of the month which is at index 0
            open_v.append(float(row[1]))  # Convert to float for more precision 
            #high_v.append(float(row[2]))
            #low_v.append(float(row[3]))
            #close_v.append(float(row[4]))
            #adj_v.append(float(row[5]))
            #v.append(int(row[6]))
        return

def predict_price(dates, prices, x):

    reshaped_dates = np.reshape(dates, len(dates), 1)  # converting to matrix of n X 1
    #dates = dates.reshape(1, -1)
    #svr_lin1 = SVR(kernel='linear', C=1e3)  # 1e3 denotes 1000
    svr_lin2 = SVR(kernel='rbf', C=1e3)
    #svr_lin3 = SVR(kernel='poly', C=1e3)
    #svr_lin1.fit(reshaped_dates, prices)
    svr_lin2.fit(dates, prices)
    #svr_lin3.fit(dates, prices)
    
    # This plots the initial data points as black dots with the data label and plot
    # each of our models as well

    plt.scatter(reshaped_dates, open_v, color='black', label='Data')  
    
    #plt.plot(open_v, svr_lin1.predict(open_v), color='red')  # plotting the line made by linear kernel
    plt.plot(open_v, svr_lin2.predict(open_v), color='blue')  # plotting the line made by linear kernel
    #plt.plot(open_v, svr_lin3.predict(open_v), color='green')  # plotting the line made by linear kernel
        
    plt.xlabel('Date')  # Setting the x-axis
    plt.ylabel('Price')  # Setting the y-axis
    plt.title('Support Vector Regression')  # Setting title
    plt.legend()  # Add legend
    plt.show()  # To display result on screen

    return svr_lin2.predict(x)[0]

get_data('C://Users//theabhishekg//Desktop//ML FIN//YF.csv') 
predicted_price = predict_price(dates, open_v, 29)

print('The predicted prices are:', predicted_price)
