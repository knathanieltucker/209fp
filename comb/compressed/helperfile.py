from numpy import nan
from numpy.random import randn
import numpy as np
from pandas.core.common import adjoin
from pandas import *
from pandas.io.data import DataReader
import pandas.util.testing as tm
tm.N = 10
import scikits.statsmodels.api as sm
import scikits.statsmodels.datasets as datasets
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import scipy.stats as stats
import matplotlib.pyplot as plt

def rolling_betas(returns, benchmark, window=250):
    # SPX betas
    betas = {}
    for col, ts in returns.iteritems():
        betas[col] = _rolling_beta(ts, spx, window=250)
    return DataFrame(betas)

def leave_one_out(data, yvar='AAPL'):
    data = data.copy()
    y = data.pop(yvar)
    result = {}
    for var in data.columns:
        X = data.drop([var], axis=1)
        model = ols(y=y, x=X)
        result['ex-%s' % var] = model.beta
    return result


def side_by_side(*objs, **kwds):
    space = kwds.get('space', 4)
    reprs = [repr(obj).split('\n') for obj in objs]
    print adjoin(space, *reprs)

plt.rc('figure', figsize=(10, 6))
np.random.seed(123456)
panel = Panel.load('data_panel')
close_px = panel['Adj Close']
RR = close_px / close_px.shift(1) - 1
index = (1 + RR).cumprod()
eom_index = index.asfreq('EOM')
eom_index = index.asfreq('EOM', method='ffill')


def plot_returns(port_returns, bmk_returns):
    plt.figure()
    cum_port = ((1 + port_returns).cumprod() - 1)
    cum_bmk = ((1 + bmk_returns).cumprod() - 1)
    cum_port.plot(label='Portfolio returns')
    cum_bmk.plot(label='Benchmark')
    plt.title('Portfolio performance')
    plt.legend(loc='best')
def calc_te(weights, univ_RR, track_RR):
    port_RR = (univ_RR * weights).sum(1)
    return (port_RR - track_RR).std()




def plot_corr(rrcorr, title=None, normcolor=False):
    xnames = ynames = list(rrcorr.index)
    nvars = len(xnames)
    if title is None:
        title = 'Correlation Matrix'
    if normcolor:
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = None, None
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axim = ax.imshow(rrcorr.values, cmap=plt.cm.jet,
                     interpolation='nearest',
                     extent=(0,nvars,0,nvars), vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(nvars)+0.5)
    ax.set_yticklabels(ynames[::-1], fontsize='small',
                       horizontalalignment='right')
    ax.set_xticks(np.arange(nvars)+0.5)
    ax.set_xticklabels(xnames, fontsize='small',rotation=45,
                       horizontalalignment='right')
    plt.setp( ax.get_xticklabels(), fontsize='small', rotation=45,
             horizontalalignment='right')
    fig.colorbar(axim)
    ax.set_title(title)

def plot_acf_multiple(ys, lags=20):
    """

    """
    from scikits.statsmodels.tsa.stattools import acf
    old_size = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 8
    plt.figure(figsize=(10, 10))
    xs = np.arange(lags + 1)
    acorr = np.apply_along_axis(lambda x: acf(x, nlags=lags), 0, ys)
    k = acorr.shape[1]
    for i in range(k):
        ax = plt.subplot(k, 1, i + 1)
        ax.vlines(xs, [0], acorr[:, i])
        ax.axhline(0, color='k')
        ax.set_ylim([-1, 1])
        ax.set_xlim([-1, xs[-1] + 1])
    mpl.rcParams['font.size'] = old_size

def load_mplrc():
    import re
    path = 'matplotlibrc'
    regex = re.compile('(.*)[\s]+:[\s]+(.*)[\s]+#')
    for line in open(path):
        m = regex.match(line)
        if not m:
            continue
        cat, attr = m.group(1).strip().rsplit('.', 1)
        plt.rc(cat, **{attr : m.group(2).strip()})

		
def rolling_betas(returns, benchmark, window=250):
    betas = {}
    for col, ts in returns.iteritems():
        betas[col] = _rolling_beta(ts, spx, window=250)
    return DataFrame(betas)
def _rolling_beta(returns, benchmark, window=250):
    model = ols(y=returns, x={'bmk' : benchmark}, window=window,
                min_periods=100)
    return model.beta['bmk']
	
def get_monthly_returns(daily_returns):
    daily_index = to_index(daily_returns)
    monthly_index = daily_index.asfreq('EOM', method='pad')
    return to_returns(monthly_index)
	
def to_index(returns):
    return (1 + returns).cumprod()
def to_returns(prices):
    return prices / prices.shift(1) - 1
		
		
def get_opt_holdings_opt(univ_RR, track_RR):
    import scipy.optimize as opt
    K = len(univ_RR.columns)
    init_weights = np.ones(K) / K
    result = opt.fmin_l_bfgs_b(calc_te, init_weights,
                               args=(univ_RR.values,
                                     track_RR.values),
                               approx_grad=True, factr=1e7, pgtol=1e-7)
    return Series(result[0], index=univ_RR.columns)
