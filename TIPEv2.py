import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd 


AAPL = pd.read_csv('AAPL.csv')
AMZN = pd.read_csv('AMZN.csv')
FB = pd.read_csv('FB.csv')
GOOG = pd.read_csv('GOOG.csv')
TSLA = pd.read_csv('TSLA.csv')
NFLX = pd.read_csv('NFLX.csv')


c = 'Close'

df = [AAPL['Date'],AAPL[c],AMZN[c],FB[c],GOOG[c],TSLA[c],NFLX[c]]
code_tickers = ['Date', 'AAPL','AMZN','FB','GOOG','TSLA','NFLX']


dic = {} 

for i in range(len(code_tickers)):
	dic[code_tickers[i]] = list(df[i])

market_data_frame = pd.DataFrame(dic)
market_data_frame = market_data_frame.set_index('Date')



def statistical_estimation(market_data_frame):
#Function to estimate empirically the mean and covariance of the vector Y
	hist_y = (market_data_frame.copy()).iloc[:-1, :]  #removing the data of last day
	for i in range(len(hist_y.index)):
		date = market_data_frame.index[i]
		next_date = market_data_frame.index[i+1]
		hist_y.loc[date] = market_data_frame.loc[next_date]/market_data_frame.loc[date]
	return hist_y.mean() , hist_y.cov()


class Market:
    #Class regrouping parameters of the market
    def __init__(self, r, mu, omega, date):
        self.p0 = p0
        self.r = r
        self.mu = mu
        self.omega = omega

N = len(code_tickers) - 1  #remove 'Date'
p0 = market_data_frame.iloc[-1, :]	# last day in our data , which is also the first day in our model
r = 0.0001
mu, omega = statistical_estimation(market_data_frame)
market = Market(r, mu, omega, p0)


class Portfolio:
    #Class modelizing the portfolio 
    def __init__(self, a0, a, name=""):
        self.a0 = a0
        self.a = a
        self.name = name
    
    def mean(self, market):
        wa = np.dot(np.diag(market.p0), self.a)
        return self.a0*(1+market.r) + np.dot(wa.transpose(), mu)
    
    def variance(self, market):
        wa = np.dot(np.diag(market.p0), self.a)
        return np.dot(wa.transpose(), np.dot(market.omega, wa))


def optimal_portfolio(market, v, sigma):
    #returns the optimal porfolio according to markowitz's model
	N = len(market.mu)
	mu_tilde = market.mu - (1+r)
	if sigma > 0:
		lambda_star = sqrt(np.dot(mu_tilde.transpose(),
		np.dot(np.linalg.inv(omega), mu_tilde)))/sigma
		wa = (1/lambda_star)*np.dot(np.linalg.inv(omega), mu_tilde)
		a = np.dot(np.linalg.inv(np.diag(market.p0)), wa)
		e = np.ones(N)
		a0 = v - np.dot(wa.transpose(), e)
	else:
		a = np.zeros(N)
		a0 = v
	return Portfolio(a0, a)  

def efficient_frontier(market, v, sigma_range):
    # Preparing the plot in the mean variance space
	efficients = []
	for sigma in sigma_range:
		portfolio = optimal_portfolio(market, v, sigma)
		efficients.append(portfolio.mean(market))
	return efficients

def plot_portfolios(v, market):
    """
    ploting the efficient frontier and portfolios invested only in some chosen stocks
    in addition to some particular portfolios : 100% invested in risky asset and 100% invested in non
    risky asset
    
    """
    I = np.eye(N)
    unique_stock_portfolios = []
    for i,code in enumerate(code_tickers[1:]):
        a = (v/(market.p0[i]))*I[i,:]
        portfolio = Portfolio(0, a, code)	
        unique_stock_portfolios.append(portfolio)
    min_sigma = 0
    min_return = v*(1+market.r)
    mu_tilde = market.mu - (1+market.r)
    e = np.ones(len(market.mu))
    sigma_star = v*sqrt(np.dot(mu_tilde.transpose(),
    np.dot(np.linalg.inv(market.omega),
    mu_tilde)))/np.dot(mu_tilde.transpose(),
    np.dot(np.linalg.inv(market.omega), e))
    mean_star = optimal_portfolio(market, v, sigma_star).mean(market)

    fig, ax = plt.subplots(figsize=(15, 10))
    colors = ['b','c', 'm', 'y', 'k', 'darkorange']

    sigma_max = 0
    for i, portfolio in enumerate(unique_stock_portfolios):
        sigma = sqrt(portfolio.variance(market))
        if sigma > sigma_max:
            sigma_max = sigma
        mean = portfolio.mean(market)
        ax.scatter(sigma,mean,marker="P",color=colors[i],s=500,label=portfolio.name)
#sigma_max = max(sigma_max, sigma_star)
    sigma_range = np.linspace(0, 2*sigma_star, num=10)
    means = efficient_frontier(market, v, sigma_range)    
    ax.scatter(sigma_star,mean_star,marker='*',color='r',s=500,
           label='Portefeuille $P^*$ 100% actif risqué')
    ax.scatter(min_sigma,min_return,marker='*',color='g',s=500,
           label='Portefeuille $P^0$ 100% actif non risqué')
    ax.vlines(sigma_star, 0.995*min(means), 1.005*max(means),
          colors='r', linestyles='dashed')
    ax.plot(sigma_range, means, linestyle='-.', color='black',
        label='Frontière efficiente')
    ax.set_title("Positionnement des portefeuilles dans l'espace moyenne variance")
    ax.set_xlabel('$\sigma$')
    ax.set_ylabel('$E[V_1]$')
    ax.legend(labelspacing=0.8)
    plt.savefig("plot.png")
    
v = 10
plot_portfolios(v, market)



