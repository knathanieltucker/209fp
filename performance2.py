import pandas as pd
import numpy as np
import ystockquote as ysq



def check_performance(data):


    performancedictlist = []
    counter = 0
    for throw,frame in data.iterrows():
        performancedictlist.append(ysq.get_historical_prices(frame['company'],frame['date'],frame['date']).values()[0])
        counter += 1
        if counter % 25 == 0:
            print counter
    performanceval = np.array([float(stock['Open'])-float(stock['Close']) for stock in performancedictlist])
    performance = np.divide(performanceval,np.abs(performanceval))/2+.5
    data['performance'] = pd.Series(performance,index = data.index)
    return data,performance

def check_performance_scaled(data):


    performancedictlist = []
    counter = 0
    for throw,frame in data.iterrows():
        performancedictlist.append(ysq.get_historical_prices(frame['company'],frame['date'],frame['date']).values()[0])
        counter += 1
        if counter % 25 == 0:
            print counter
    performanceval = np.array([float(stock['Open'])-float(stock['Close']) for stock in performancedictlist])
    performance = np.clip(np.divide(performanceval,np.array([float(stock['Open']) for stock in performancedictlist])),-1,1)/2.+0.5
    data['performance'] = pd.Series(performance,index = data.index)
    return data,performance
