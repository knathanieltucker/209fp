
"""
a set of useful indicators 
date started: June 7 2013
date last edited: June 7 2013
author: Nathaniel Tucker 
email: tucker@college.harvard.edu
"""



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
	up = []
	down = []
	for i in range(1,len(input)):
		if input[i]>input[i-1]:
			#current - previous
			up.append(input[i]-input[i-1])
			down.append(0)
		else:
			#previous - current
			down.append(input[i-1]-input[i])
			up.append(0)


	#step 2:
	# find the Relative Strength, RS, of the ups and downs and then divide 
	def rs(diff):
		out = [sum(diff[0:period])/period]
		for i in range(period,len(diff)):
			out.append((diff[i] + (period-1)*out[i-period]) /float(period))
		return out

	rs_up = rs(up)
	rs_down = rs(down)

	#a deviding function to aviod devide by 0 
	def dev(a,b):
		if b == 0: return 1
		return a/b
	
	relative_strength = map(lambda x: dev(x[0],x[1]), zip(rs_up,rs_down))

	#step 3 
	#RSI normalization  
	return map(lambda x: 100 - (100/(1+x)),relative_strength)

