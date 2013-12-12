
"""
a set of time series analysis filters 
date started: January 6 2013
date last edited: June 7 2013
author: Nathaniel Tucker 
email: tucker@college.harvard.edu
"""


"""
A first order simple moving average 
https://en.wikipedia.org/wiki/Moving_average
input is a list of floats and period is an int
"""
def sma(input, period):
	if len(input) < period : raise NameError("sma: list length less than the period")
    	av = [0 for i in range(period)]
	for day in range(period,len(input)):
		av.append(sum(input[day - period:day])/period)
	return av



"""second order exponential moiving average, gaussian filter returns a list (the filtered signal)
http://www.mesasoftware.com/Papers/SWISS%20ARMY%20KNIFE%20INDICATOR.pdf
http://en.wikipedia.org/wiki/Gaussian_filter
input is a list of length 3 or greater
check out http://www.mesasoftware.com/Papers/SWISS%20ARMY%20KNIFE%20INDICATOR.pdf
"""
def gaussian(input,alpha):
	if len(input) < 3 : raise NameError("gaussian_filter: list length less than 3")
	av = sum(input)/float(len(input))
	output = [av+alpha,av+alpha*2] 
	for i in range(2,len(input)):
		output.append( ((alpha**2) * input[i]) + (2*(1-alpha)*output[i-1]) -  (((1-alpha)**2) * output[i-2]))
	return output

""" exponential moving average 
http://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
input of the list must be not fewer elements than period
"""
def ema(input, period):
	if len(input)<period: raise NameError("ema: list was smaller than the period")
	alpha = 2.0 / (float(period)+1.0)
	output = [sum(input[0:period])/period]
	for i in range(1,len(input)):
		output.append(alpha * input[i] + (1-alpha)*output[i-1])
	return output


"""butterworth filter -- very smooth 
takes and alpha and a list, creates a list 
check out http://www.mesasoftware.com/Papers/SWISS%20ARMY%20KNIFE%20INDICATOR.pdf
list must be at least 3 elements long
"""
def butterworth(input, alpha):
	if len(input) < 3 : raise NameError("butterworth: list length less than 3")
	av = sum(input)/float(len(input))
	output = [av+alpha,av+alpha*2] 
	for i in range(2,len(input)):
		ins=((alpha**2) * input[i])/4 + 2*input[i-1] + input[i-2]
		outs=2*(1-alpha)*output[i-1] -  ((1-alpha)**2) * output[i-2]
		output.append(ins+outs)
	return output




