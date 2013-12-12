
from pandas import *
import pandas.io.data as dt
reload(dt)
from pandas.io.data import DataReader

indexes = {'SPX' : '^GSPC'}
stocks = ['AAPL', 'GE', 'IBM', 'MSFT', 'XOM', 'AA', 'JNJ', 'PEP']


start = datetime(1990, 1, 1)
end = datetime.today()

data = {}
for stock in stocks:
    print stock
    stkd = DataReader(stock, 'yahoo', start, end).sort_index()
    data[stock] = stkd

for name, ticker in indexes.iteritems():
    print name
    stkd = DataReader(ticker, 'yahoo', start, end).sort_index()
    data[name] = stkd

# data = Panel(data).swapaxes(0, 2)
# data.save('data_panel')
