"""
author: Nathaniel Tucker 
email: tucker@college.harvard.edu
title: signal processing 
"""



#below we have varius filters used in signal processing

"""
A first order simple moving average 
https://en.wikipedia.org/wiki/Moving_average
time_series is a list of floats and period is an int
"""
def sma(time_series, period):
    if len(time_series) < period : raise NameError("sma: list length less than the period")
    av = [time_series[0] for i in range(period)]
    for day in range(period,len(time_series)):
		av.append(sum(time_series[day - period:day])/period)
    return av



"""second order exponential moiving average, gaussian filter returns a list (the filtered signal)
http://www.mesasoftware.com/Papers/SWISS%20ARMY%20KNIFE%20INDICATOR.pdf
http://en.wikipedia.org/wiki/Gaussian_filter
time_series is a list of length 3 or greater
check out http://www.mesasoftware.com/Papers/SWISS%20ARMY%20KNIFE%20INDICATOR.pdf
"""
def gaussian(time_series,period):
	if len(time_series) < 3 : raise NameError("gaussian_filter: list length less than 3")
	alpha = 2.0 / (float(period)+1.0)
	av = sum(time_series)/float(len(time_series))
	output = [av+alpha,av+alpha*2] 
	for i in range(2,len(time_series)):
		output.append( ((alpha**2) * time_series[i]) + (2*(1-alpha)*output[i-1]) -  (((1-alpha)**2) * output[i-2]))
	return output

""" exponential moving average 
http://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
time_series of the list must be not fewer elements than period
"""
def ema(time_series, period):
	if len(time_series)<period: raise NameError("ema: list was smaller than the period")
	alpha = 2.0 / (float(period)+1.0)
	output = [sum(time_series[0:period])/period]
	for i in range(1,len(time_series)):
		output.append(alpha * time_series[i] + (1-alpha)*output[i-1])
	return output


"""butterworth filter -- very smooth 
takes and alpha and a list, creates a list 
check out http://www.mesasoftware.com/Papers/SWISS%20ARMY%20KNIFE%20INDICATOR.pdf
list must be at least 3 elements long
"""
def butterworth(time_series, period):
	if len(time_series) < 3 : raise NameError("butterworth: list length less than 3")
	alpha = 2.0 / (float(period)+1.0)
	av = sum(time_series)/float(len(time_series))
	output = [av+alpha,av+alpha*2] 
	for i in range(2,len(time_series)):
		ins=((alpha**2) * time_series[i])/4 + 2*time_series[i-1] + time_series[i-2]
		outs=2*(1-alpha)*output[i-1] -  ((1-alpha)**2) * output[i-2]
		output.append(ins+outs)
	return output

"""RSI(x) relative strength indicator 
	step 1: find the change in the measured 
	step 2: then find the relative strengths 
	step 3: then normalize with 100 - (100/(1+data))
	will take period and a list and ouput a list, list must be longer than period 
"""
def rsi(input,period):
	# step 1: find the day by day change in the measured quntities 
	# and put them into a list of ups and downs 
	if len(input)<period: raise NameError("rsi: list was smaller than the period")
	up = [0]
	down = [0]
	for i in range(1,len(input)):
		if input[i]>input[i-1]:
			#current - previous
			up.append(input[i]-input[i-1])
			down.append(0)
		else:
			#previous - current
			down.append(input[i-1]-input[i])
			up.append(0)


	rs_up = sma(up,period)
	rs_down = sma(down,period)

	#a deviding function to aviod devide by 0 
	def dev(a,b):
		if b == 0: return 1
		return a/b
	
	relative_strength = map(lambda x: dev(x[0],x[1]), zip(rs_up,rs_down))

	#step 3 
	#RSI normalization  
	return map(lambda x: 100 - (100/(1+x)),relative_strength)




# below is a set of funcitons used to generate equity

"""get the day by day return of a protfolio of 100 dollars 
given two data files: an indicator and a verification 
consisting of the same number of elements, over the same dates**
one file will serve as indicator and the other will serve as verification,

indicator == a series of floats above 0

verification == a series of floats above 0 

hueristic == construct a st signal_processor and a lt signal_processor from indicator, 
then follow the Value or the Technical approach, if strat is 
true, then will follow the value approach, and if false the technical 

takes float list, float list, int, int, int
returns float list 
"""
def dmac(indicator, verification,signal_processor, st, lt, holding_period, strat,end_value=False):
	#check for valid inputs 
	l = len(indicator)
	if l!=len(verification): 
		raise NameError("getreturn: indicator & verification length not equal")

	#create signals  
	st = signal_processor(indicator,st)
	lt = signal_processor(indicator,lt)

	#determine starting conditions 
	beginV = len(lt) - l
	beginS = len(lt) - len(st)

	if st[beginS] < lt[0]:
		below = True
	else:
		below = False



	money = 100.0
	returns = []	
	i = 0
	while (i + holding_period < l):
		#if the signals have crossed
		if (below and st[beginS+i]>lt[i]) or (not below and st[beginS+i]<lt[i]): 
			initial = money
			shares = money/verification[i]
			
			#start your investment
			for j in range(holding_period+1):
				#indicator is below so long
				if below ^ strat:
					gain = shares * (verification[j+i] - verification[i] )
				else: # short 
					gain = shares * verification[i] - shares * verification[i+j] 
				money = initial + gain
				returns.append(money)
			i = holding_period + i + 1
			below = not below
		else:
			returns.append(money)
			i = 1 + i
	
	if end_value:
		return money

	return (returns + [money]*holding_period)[:len(st)]



"""get the day by day return of a protfolio of 100,000 dollars 
given two data files consisting of the same number of elements, over the same dates**
one file will serve as indicator and the other will serve as verification,

indicator == a series of floats that are bounded between 0 and 100, 
if they move above 100-bounds then it indicates a short 
if they move below bounds then it indicates a long 

verification == a series of floats above 0 

hueristic == on a short/long indicator, we will look 'holding_period' days ahead, 
and then cash out then with either a gain or a loss 

takes float list, float list, int, int 
returns float list 
"""
def equity_curve(indicator, verification, bound_up, bound_low, holding_period,end_value=False):
	#check for valid inputs 
	l = len(indicator)
	if l!=len(verification) or l != len(bound_up) or l != len(bound_low): 
		raise NameError("getreturn: indicator & verification length not equal")

	money = 100.0
	returns = []	
	i = 0
	while (i + holding_period < l):
		#if the signal is out of bounds
		if indicator[i] > bound_up[i] or indicator[i] < bound_low[i]: 
			initial = money
			shares = money/verification[i]
			
			for j in range(holding_period+1):
				#indicator is below so long
				if indicator[i] > bound_up[i]:
					gain = shares * verification[j+i] - shares * verification[i] 
				else: # short 
					gain = shares * verification[i] - shares * verification[i+j] 
				money = initial + gain
				returns.append(money)
			i = holding_period + i + 1
		else:
			returns.append(money)
			i = 1 + i
			
	if end_value:
		return money

	return (returns + [money]*holding_period)[:l]




# below is a set of statistics that we will want to calculate 

from numpy import var, cov, corrcoef, polyfit, poly1d, std


"""Sortino ratio, takes a yearly of returns and a risk free rate
must have negative returns 
returns a float 
"""
def sortino_ratio(returns, Rf):
    #reduce the returns by the risk free rate 
    adjusted_returns = map(lambda x: x-Rf, returns)
    #find only the negative returns 
    def kill_neg(x):
        if x<0: return x
        return 0
    negative_adjusted_returns = map(kill_neg,adjusted_returns)
    ave = sum(adjusted_returns)/len(adjusted_returns)
    #average expected returns / standard deviation
    return ave / std(negative_adjusted_returns)

"""the sharp ratio 
takes in a yearly float return curve and a float for the risk free rate of the time period
output is a float
"""
def sharp_ratio(returns, Rf):
    #reduce the returns by the risk free rate 
    adjusted_returns = map(lambda x: x-Rf, returns)
    ave = sum(adjusted_returns)/len(adjusted_returns)
    #average expected returns / standard deviation
    return ave / std(adjusted_returns)

"""get returns, takes in an equity curve
outputs a percent returns curve 
"""
def get_return(equity):
	e_delta = []
	#then find the rate of return  point by point
	for i in range(len(equity)-1):
		e_delta.append(float(equity[i+1]-equity[i])/equity[i])
	return e_delta

"""r-squared, takes in two returns 
first fit points to a degree degree polynomial,
x and y must be same length 
returns a float 
"""
def r_squared(x, y, degree):
	#first check if both arrays are of equal length
	l = len(x)
	if l!=len(y): raise NameError("r_squared: x & y length not equal")

	coeffs = polyfit(x, y, degree)

	# r-squared
	p = poly1d(coeffs)
	# fit values, and mean
	yhat = [p(z) for z in x]
	ybar = sum(y)/len(y)
	ssreg = sum([ (yihat - ybar)**2 for yihat in yhat])
	sstot = sum([ (yi - ybar)**2 for yi in y])
	return ssreg / sstot


"""Alpha, find the alpha of given strategy 
Rp = portfolio return, Rf = risk free rate, Rm = market return, b = beta 
takes in all floats and returns a float 
"""
def alpha(Rp, Rf, Rm, b):
	#just a formula
	return Rp-(Rf + (Rm - Rf)*b)

"""Find the correlation between two returns  
lists must be the same length 
returns a float 
"""
def correlation(x,y):
	#first check if both arrays are of equal length
	l = len(x)
	if l!=len(y): raise NameError("correlation: x & y length not equal")

	#use numpy to find the correlation  
	return corrcoef(y,x)[0][1]

"""Find the beta of actual comparted to a benchmark returns 
inputs must be same length,
first will find the rate of returns for both assets, then calculate beta using cov/var numpy 
will output a float 
"""
def beta(benchmark, actual):
	#first check if both arrays are of equal length
	l = len(actual)
	if l!=len(benchmark): 
		raise NameError("beta: benchmark & actual length not equal")

	#beta formula cov(bench,actual)/var(bench)
	return (cov(benchmark,actual, ddof=0)[0][1])/var(benchmark)


# this function sums up the others and returns a 
# group of stats 

""" performance: find the performance of data and return in a dict
	the stats:
	1: year to date return = ytd 
	2: annualized return  = aret 
	3: 1 year compound annual return = car1
	4: 3 year compound annual return = car3 
	5: 5 year compound annual return = car5 
	6: cummulative return = cret
	7: average monthly = ave
	8: value of 1000 = 1000
	9: positive months = pos 
	10: max drawdown = max
	11: sharp_ratio = sharp
	12: sortino ratio = sortino 
takes in actual equity curve, in month periods  
""" 
def performance(equity,time_frame=12, Rf = .0005):
	l = len(equity)

	# 1: ytd return
	per = {}
	ytd = (equity[l-1] - equity[l-1-time_frame])/equity[l-1-time_frame]
	per['ytd'] = ytd*100

	# 2: annualized return, or Cagr formula 
	per['aret'] = 100*((equity[l-1]/equity[0])**(1.0/(l/time_frame)) - 1)

	# 3-5 cars 
	per['car1'] = 100*((equity[l-1]/equity[l-1-time_frame])**(1.0/1) - 1)
	per['car3'] = 100*((equity[l-1]/equity[l-1-3*time_frame])**(1.0/3) - 1)
	per['car5'] = 100*((equity[l-1]/equity[l-1*5*time_frame])**(1.0/5) - 1)

	#: 6 cummulative return 
	per['cret'] = 100*((equity[l-1] - equity[0])/equity[0])

	# 7: average month value 
	returns = get_return(equity)
	per['ave'] = 100*(sum(returns)/len(returns))

	# 8: value of 1000
	per['1000'] = 1000*equity[l-1]/equity[0]

	# 9: percent positive months 
	per['pos'] = float(reduce(lambda x,y: x + int(y>0),returns,0))/l

	# 10: drawdown 
	min_ = 0.0
	drawdowns = []
	for i in range(l-1):
		if equity[i] > equity[i+1]:
			min_ = equity[i]
			for j in range(l-i-1):
				if equity[j+i] < min_:
					min_ = equity[j+i]
				if equity[j+i] > equity[i]:
					drawdowns.append((equity[i]-min_)/equity[i])
					break
		

	per['max'] = 100*max(drawdowns)

	# 11: sharp, but first must construct a yearly return curve
	yearly = []
	for i in range(len(equity)):
		if (i)%time_frame == 0:
			yearly.append(equity[i])	

	yearly = get_return(yearly)

	per['sharp'] = sharp_ratio(yearly, Rf)

	# 12: sortino 
	per['sortino'] = sortino_ratio(yearly, Rf)

	#standard deviation 
	per['std'] = std(yearly)

	return per

""" stats: find the stats of data and return in a dict
	the stats:
	1: final return = return 
	2: beta against benchmark = beta 
	3: correlation against benchmark = correlation 
	4: alpha against benchmark = alpha
	5: r-squared against benchmark = r
takes in actual equity curve, benchmark ec, and risk free rate 
""" 
def stats(actual, benchmark, time_frame=12, Rf= .0005, degree = 1):
	l = len(actual)
	if l!=len(benchmark): 
		raise NameError("stats: benchmark & actual length not equal")

	# 1: final return, last - first / first 
	stat = {}
	Rp = (actual[l-1] - actual[0])/actual[0]
	stat['return'] = Rp * 100
	

	# 2: return for benchmark and actual then beta 
	returns = get_return(actual)
	returns_b = get_return(benchmark)

	b = beta(returns,returns_b)
	stat['beta'] = b

	# 3: correlation 
	stat['correlation'] = correlation(returns,returns_b)


	# 4: find benchmark tot, and annuallize return then alpha 
	Rm = (benchmark[l-1]/benchmark[0])**(1.0/(l/time_frame)) - 1
	if Rp < 0:
		stat['alpha'] = 0
	else:
		Rp = Rp**(1.0/(l/time_frame)) - 1
		stat['alpha'] = 100* alpha(Rp, Rf, Rm, b)

	

	# 5: r-squared, default is degree 1 poly fit 
	stat['r'] = r_squared(returns, returns_b, degree) 

	return stat


# and finally an example using zipline 

import matplotlib.pyplot as plt
import pandas as pd

from zipline.algorithm import TradingAlgorithm
import zipline.finance.trading as trading
from zipline.transforms import MovingAverage
from zipline.utils.factory import load_from_yahoo

from datetime import datetime
import pytz


class DualMovingAverage(TradingAlgorithm):
    """Dual Moving Average Crossover algorithm.

    This algorithm buys apple once its short moving average crosses
    its long moving average (indicating upwards momentum) and sells
    its shares once the averages cross again (indicating downwards
    momentum).

    """
    def initialize(self, short_window=20, long_window=40):
        # Add 2 mavg transforms, one with a long window, one
        # with a short window.
        self.add_transform(MovingAverage, 'short_mavg', ['price'],
                           window_length=short_window)

        self.add_transform(MovingAverage, 'long_mavg', ['price'],
                           window_length=long_window)

        # To keep track of whether we invested in the stock or not
        self.invested = False

    def handle_data(self, data):
        self.short_mavg = data['IBM'].short_mavg['price']
        self.long_mavg = data['IBM'].long_mavg['price']
        self.buy = False
        self.sell = False

        if self.short_mavg > self.long_mavg and not self.invested:
            self.order('IBM', 5000)
            self.invested = True
            self.buy = True
        elif self.short_mavg < self.long_mavg and self.invested:
            self.order('IBM', -5000)
            self.invested = False
            self.sell = True

        self.record(short_mavg=self.short_mavg,
                    long_mavg=self.long_mavg,
                    buy=self.buy,
                    sell=self.sell)

def example():
    start = datetime(1990, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(1991, 1, 1, 0, 0, 0, 0, pytz.utc)
    data = load_from_yahoo(stocks=['IBM'], indexes={}, start=start,
                           end=end)

    dma = DualMovingAverage()
    results = dma.run(data)

    index = [br.date for br in trading.environment.benchmark_returns]
    rets = [br.returns for br in trading.environment.benchmark_returns]
    bm_returns = pd.Series(rets, index=index).ix[start:end]
    results['benchmark_returns'] = (1 + bm_returns).cumprod().values
    results['algorithm_returns'] = (1 + results.returns).cumprod()
    fig = plt.figure()
    ax1 = fig.add_subplot(211, ylabel='cumulative returns')

    results[['algorithm_returns', 'benchmark_returns']].plot(ax=ax1,
                                                             sharex=True)

    ax2 = fig.add_subplot(212)
    data['IBM'].plot(ax=ax2, color='r')
    results[['short_mavg', 'long_mavg']].plot(ax=ax2)

    ax2.plot(results.ix[results.buy].index, results.short_mavg[results.buy],
             '^', markersize=10, color='m')
    ax2.plot(results.ix[results.sell].index, results.short_mavg[results.sell],
             'v', markersize=10, color='k')
    plt.legend(loc=0)

    # sharpe = [risk['sharpe'] for risk in dma.risk_report['one_month']]
    # print "Monthly Sharpe ratios:", sharpe









