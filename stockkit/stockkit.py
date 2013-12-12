
"""
a set of useful functions that can be used in stock analysis 
date started: January 6 2013
date last edited: January 22 2013
author: Nathaniel Tucker 
email: tucker@college.harvard.edu
"""

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





