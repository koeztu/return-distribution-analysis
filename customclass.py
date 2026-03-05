


import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.stats import skew
from scipy.stats import tmean
from scipy.stats import tstd
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm
import random
from joblib import Parallel, delayed



class BootstrapClass:
    """
    A custom class for the bootstrap analysis.
    Includes:
    - Functions to compute portfolio weights
    - Parallelized portfolio bootstrapping functionality
    - Regression analysis functionality

    """

    def __init__(self, return_data_path="./DATA/ret_full.csv", size_data_path="./DATA/sz_full.csv", sector=0, pf_sizes=np.arange(5, 50 + 1, 5), pf_horizons=np.arange(3, 25 + 1, 2), n_outcomes=200):
        """
        Initialization. Simply computes and sets class variables.

        """

        # read data from file
        ret_df = pd.read_csv(return_data_path, header=0, index_col=0)
        sz_df = pd.read_csv(size_data_path, header=0, index_col=0)

        # convert index to datetime format
        ret_df.index = pd.to_datetime(ret_df.index)

        # get indices of the first days of each month
        self.months = ret_df.index.month.to_numpy()
        self.rebalancing_days = list(np.where(self.months[1:] != self.months[:-1])[0] + 1) # first of month indices

        # convert to numpy array for faster computations
        self.ret = ret_df.to_numpy()
        self.sz = sz_df.to_numpy()

        # set size and horizon class variables as specified in the constructor call
        self.pf_sizes = pf_sizes
        self.pf_horizons = pf_horizons # investment horizon in years
        self.pf_horizons = self.pf_horizons * 12 # investment horizon in months

        # compute number l of bootstrapped datapoints per method
        self.l = len(self.pf_sizes) * len(self.pf_horizons)

        # create lists that form the size and horizon columns for the regression data table
        self.sizes = []
        self.horizons = []

        for size in self.pf_sizes:
            for horizon in self.pf_horizons:
                self.sizes.append(size)
                self.horizons.append(horizon / 12) # in years


        # set the index where the bootstrap starts
        # we need 1 year, i.e. 12 months of history to compute the covariance matrix
        self.bootstrap_start = 12 # position of first bootstrap day in rebalancing_days list

        # set a holding period of 3 months (interval between rebalancing or re-optimization)
        self.holding_period = 3

        # set the number of long-run outcomes to compute per skew estimation as specified in constructor call (a higher number improves the estimate at the cost of computing time)
        self.n_outcomes = n_outcomes

        # get the number of stocks and trading days in the dataset
        self.n_stocks = self.ret.shape[1]
        self.n_days = self.ret.shape[0]

        # set the sector as specified in the constructor
        self.sector = sector 

        # for each rebalancing or re-optimization point, find the stocks whose history is long enough for the covariance estimation
        self.min_valid = self.n_stocks + 1 # the minimum number of valid stocks ever reached
        self.valid_stocks = []

        # before the start, no stocks are valid
        for i in range(self.bootstrap_start):
            self.valid_stocks.append([])

        for index, day in enumerate(self.rebalancing_days[self.bootstrap_start:-self.holding_period], start=self.bootstrap_start):
            valid_stocks_temp = []

            for stock in range(self.n_stocks):

                #the window for covariance estimation is [start_day, day]
                start_day = self.rebalancing_days[index - self.bootstrap_start]

                if np.sum(np.isnan(self.ret[start_day:day, stock])) == 0: # if returns are available
                    if np.sum(np.isnan(self.sz[day, stock])) == 0: # and if market cap is available
                        valid_stocks_temp.append(stock) # add the stock to the valid list

            # update the minimum
            self.valid_stocks.append(valid_stocks_temp)
            if len(valid_stocks_temp) < self.min_valid:
                self.min_valid = len(valid_stocks_temp)



        # pad the list to match the length of rebalancing_days
        for i in range(self.holding_period):
            self.valid_stocks.append([])
        assert len(self.valid_stocks) == len(self.rebalancing_days) #if this is not true there was an error

        # make sure that the minimum number of valid stocks is higher than the maximum portfolio size
        if self.min_valid <= np.max(self.pf_sizes): 
            print(f"ATTENTION: Not enough valid stocks somewhere!") 



    def min_var_weights(self, selection, n, day, start_day):
        """
        Arguments:
        - selection: selection of stocks to be included in the portfolio
        - n: number of stocks in the portfolio
        - day: current day
        - start_day: starting day of the window for covariance estimation

        Returns: Minimum variance portfolio weights.

        """

        # set return window
        ret_window = self.ret[start_day:day, selection]

        # estimate the covariance matrix
        #cov_matrix = pd.DataFrame(ret_window).cov().values
        cov_matrix = LedoitWolf().fit(ret_window).covariance_

        L = 0.002 # regularization parameter
        eq = 1/n # equal weight target

        # define the objective function for the minimizer
        def portfolio_variance(w):
            return w @ cov_matrix @ w + L * np.sum((w - eq) ** 2)

        # provide the gradient of the objective function, this speeds up convergence of the minimization
        def gradient(w):
            return 2 * (cov_matrix @ w + L * (w - eq))
        
        # set leverage constraint: weights must sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # set long-only constraint: weights must be between 0 and 1
        bounds = [(0.0, 1.0)] * n
        
        # set initial guess: equal weight
        w0 = np.ones(n) / n

        # minimize 
        result = minimize(
            portfolio_variance,
            w0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-12}
        )

        return result.x


    def risk_parity_weights(self, selection, n, day, start_day):
        """
        Arguments:
        - selection: selection of stocks to be included in the portfolio
        - n: number of stocks in the portfolio
        - day: current day
        - start_day: starting day of the window for covariance estimation

        Returns: Risk parity portfolio weights.

        """

        # set return window
        ret_window = self.ret[start_day:day, selection]
        
        # estimate the covariance matrix
        #cov_matrix = pd.DataFrame(ret_window).cov().values
        cov_matrix = LedoitWolf().fit(ret_window).covariance_


        def spinus_algorithm(cov_matrix, n, tol=1e-9, max_iter=200):
            """
            Compute risk parity weights using Spinu's algorithm (Spinu, 2013).
            
            The goal is to solve Cx = b/x where:
            - C is the covariance matrix
            - x is the vector of portfolio weights
            - b is the vector of risk budget weights (in our case this filled with 1/n, because we want risk parity weights)

            Arguments:
            cov_matrix: covariance matrix of returns
            n: number of assets
            tol: Convergence tolerance
            max_iter: Maximum number of iterations
                
            Returns: Risk parity weights
           
            """
            
            # risk budget vector
            b = np.ones(n) / n
            
            # rescale b so that min(b) = 1
            b = b / np.min(b)
            
            # algorithm parameter
            lmbda_star = 0.95 * (3 - np.sqrt(5)) / 2  # 0.1382

            # initial guess (theorem 3.4):
            u1 = np.ones(n)
            S = np.sum(b)
            x = (np.sqrt(S) / np.sqrt(u1 @ cov_matrix @ u1)) * u1
                
            for iteration in range(max_iter):
                
                u = cov_matrix @ x - b / x
                
                H = cov_matrix + np.diag(b / (x ** 2))
                
                Delta_x = np.linalg.solve(H, u)
                
                delta = np.linalg.norm(Delta_x / x, ord=np.inf)
                
                lmbda = np.sqrt(u @ Delta_x)

                # check for convergence
                if lmbda <= tol:
                    break
                
                # update step:
                if lmbda > lmbda_star:
                    # damped phase: use smaller step to ensure we stay in valid region
                    step_size = 1.0 / (1.0 + delta)
                    x = x - step_size * Delta_x
                else:
                    # quadratic phase: use full Newton step
                    x = x - Delta_x
                
                # ensure all weights remain strictly positive
                x = np.maximum(x, 1e-10)
            
            # normalize weights to sum to 1
            w = x / np.sum(x)
            
            return w

        result = spinus_algorithm(cov_matrix, n) 
        
        return result


    def value_weights(self, selection, day):
        """
        Arguments:
        - selection: selection of stocks to be included in the portfolio
        - day: current day

        Returns: Value-weighted portfolio weights.

        """

        sz_window = self.sz[day, selection]

        assert (np.sum(np.isnan(sz_window)) == 0) # make sure there are no missing values

        # normalize to sum to 1
        w = sz_window / np.sum(sz_window)

        return w


    def random_weights(self, n):
        """
        Arguments:
        - n: number of stocks in the portfolio

        Returns: Random portfolio weights.

        """

        # draw from a uniform distribution and normalize to sum to 1
        x = np.random.uniform(low=0.0, high=1.0, size=n)
        w = x / np.sum(x)

        return w


    def set_weights(self, method, selection, n, day, start_day):
        """
        Arguments:
        - method: portfolio construction method
        - selection: selection of stocks to be included in the portfolio
        - n: number of stocks in the portfolio
        - day: current day
        - start_day: starting day of the window for covariance estimation

        Returns: Portfolio weights for the specified method.

        """

        # call the correct function to calculate the weights according to the specified method
        if method == "equal":
            return np.full(shape=n, fill_value=(1/n))
        elif method == "value":
            return self.value_weights(selection=selection, day=day)
        elif method == "minvar":
            return self.min_var_weights(selection=selection, n=n, day=day, start_day=start_day)
        elif method == "riskpar":
            return self.risk_parity_weights(selection=selection, n=n, day=day, start_day=start_day)
        elif method == "rand":
            return self.random_weights(n=n)
        elif method == "single":
            return 1
        else:
            print("Weights not specified correctly!")
            assert 1 == 0 # just error out
            



    def bootstrap_parallel(self, method, n, horizon):
        """
        Arguments:
        - method: portfolio construction method
        - n: number of stocks in the portfolio
        - horizon: the investment horizon

        Returns: Longrun skew, mean, and standard deviation (calculated from a bootstrapped distribution of outcomes) as well as the cumulative returns of each portfolio over the entire horizon

        """

        def bootstrap_portfolio():
            """
            Arguments: None

            Returns: Bootstrapped returns of a single portfolio

            """

            # initialize return array
            pf_returns_single = np.zeros(self.n_days)

            # randomly sample the starting day for the bootstrap procedure from a valid range
            start = random.sample(self.rebalancing_days[self.bootstrap_start:-(horizon + self.holding_period)], k=1)[0]

            # find the index of the starting day
            start_index = np.where(self.rebalancing_days == start)[0][0]

            # set the day the bootstrap ends
            end = self.rebalancing_days[start_index + horizon]

            # calculate all the days on which we rebalance or re-optimize the portfolio 
            valid_reb_days = self.rebalancing_days[start_index::self.holding_period]

            # randomly sample a subset of stocks
            selected_stocks = random.sample(list(np.arange(self.n_stocks)), k=n)

            # set weights to zero because we are uninvested in the beginning
            weights = np.zeros(shape=n)
            
            for day in np.arange(start, end):

                # we assume that we can trade at closing prices, more specifically; that we can calculate new portfolio weights and trade instantaneously at the exact time the market closes

                # get the returns of portfolio constituents between market close yesterday to market close today, nan free
                ret_nan_free = np.nan_to_num(self.ret[day, selected_stocks], nan=0.0) 

                # compute the day's portfolio return with weights of the previous day to avoid lookahead bias
                pf_returns_single[day] = np.dot(weights, ret_nan_free) 

                # now we can calculate new weights or float the current weights
                if day in valid_reb_days: # if we are on a rebalance day, calculate new portfolio weights

                    # get the index of the current day
                    rebalance_index = np.where(self.rebalancing_days == day)[0][0]

                    # get the index of the first day in the lookback window for the covariance estimation
                    start_index = rebalance_index - self.bootstrap_start

                    # get the exact starting day 
                    start_day = self.rebalancing_days[start_index]

                    # re-sample the random subset of stocks
                    selected_stocks = random.sample(self.valid_stocks[rebalance_index], k=n)

                    weights = self.set_weights(method=method, selection=selected_stocks, n=n, day=day, start_day=start_day) # use data up until market close today

                else: # if we are not on a rebalance day, simply float the weights according to the day's returns
                    weights *= (1 + ret_nan_free)
                    weights /= np.sum(weights)

            return pf_returns_single

        # parallelize the computation (this substantially speeds things up)
        # the basic idea is to bootstrap multiple portfolios at the same time
        pf_returns_list = Parallel(n_jobs=-1)(
            delayed(bootstrap_portfolio)() 
            for pf in range(self.n_outcomes) # the number of bootstrapped portfolios is equal to the number of outcomes
        )

        # stack results
        pf_returns = np.column_stack(pf_returns_list)

        # calculate cumulative returns
        cum_returns = np.cumprod(1.0 + pf_returns, axis=0) - 1.0
        cum_returns *= 100 # convert to percent
        cum_ret_final = cum_returns[-1, :] # get the final row, which holds the cumulative returns over the entire horizon

        # exclude any outcome that is more than 10 standard deviations from the mean
        # this is necessary because the skew estimate is very sensitive to outliers
        mean = tmean(cum_ret_final) 
        std = tstd(cum_ret_final)
        filtered_outcomes = cum_ret_final[np.abs(cum_ret_final - mean) <= (8 * std)]

        # calculate the skew estimate
        longrun_skew = skew(filtered_outcomes, bias=False)

        # calculate the mean
        longrun_mean = tmean(filtered_outcomes) #mean return outcome

        # calculate the volatility
        longrun_std = tstd(filtered_outcomes) #volatility of outcomes


        return longrun_skew, longrun_mean, longrun_std, filtered_outcomes



    def regression(self, data=None, dep_var=None, data_path=None):
        """
        Arguments: 
        - data: data to be used for the regression
        - dep_var: dependent variable for the regression
        - data_path: path to data (for cases where it is saved somewhere)

        Note: max and min size are used to run the regression for different subsets

        Returns: Nothing

        """

        data['sizes2'] = data['sizes'] * data['sizes']
        data['1/sizes'] = 1 / data['sizes']

        # center 
        data['sizes_centered'] = data['sizes'] - data['sizes'].mean()
        data['horizons_centered'] = data['horizons'] - data['horizons'].mean()
        data['sizes2_centered'] = data['sizes2'] - data['sizes2'].mean()
        data['1/sizes_centered'] = data['1/sizes'] - data['1/sizes'].mean()

        # define a rectified linear unit function for the ratio
        def ReLU(series):
            return np.maximum(0, series)

        # define the ratio
        def ratio(mean, std, skew):
            return mean / (std * (1 + ReLU(skew)))

        if dep_var == "skew":
            # set the dependent variable according to the specification in the function call
            y = data['longrun_skews']

            # set the independent variables
            # omit dummy variable 'equal' to avoid perfect multicollinearity, this makes it the reference group
            X = data[['constant', '1/sizes_centered', 'horizons_centered', 'value', 'minvar', 'riskpar', 'rand']]

        elif dep_var == "ratio":
            y = ratio(mean=data['longrun_means'], std=data['longrun_stds'], skew=data['longrun_skews'])
            X = data[['constant', 'sizes_centered', 'sizes2_centered', 'horizons_centered', 'value', 'minvar', 'riskpar', 'rand']]
        else:
            print("Forgot to specify dependent variable!")
            return 0

        # fit OLS model
        model = sm.OLS(y, X).fit()

        # write results to dataframe
        results_df = pd.DataFrame({
            'Variable': X.columns,
            'Coefficient': model.params,
            'Std Error': model.bse,
            't-value': model.tvalues,
            'p-value': model.pvalues
        })


        output_path = f'./REG_RESULTS/results_sector{int(self.sector)}_{dep_var}.csv'
        results_df.round(5).to_csv(output_path, index=False)
        print(f"Regression results written to {output_path}")

        return 0


    def simulate(self, method):
        """
        Arguments: 
        - method: portfolio construction method

        Returns: Bootstrapped skews, means, and standard deviations.

        """

        longrun_skews = []
        longrun_means = []
        longrun_stds = []

        # run the bootstrap for every size-horizon pair
        for size in self.pf_sizes:
            for horizon in self.pf_horizons:
                longrun_skew_new, longrun_mean_new, longrun_std_new, _ = self.bootstrap_parallel(method=method, n=size, horizon=horizon)

                # append results to the lists
                longrun_skews.append(longrun_skew_new)
                longrun_means.append(longrun_mean_new)
                longrun_stds.append(longrun_std_new)

        return longrun_skews, longrun_means, longrun_stds



    def get_reg_data(self):
        """
        Arguments: None

        Returns: Dataframe for regression.

        """

        # define the portfolio construction methods
        methods = ["equal", "value", "minvar", "riskpar", "rand"]
        
        dfs = [] # list of dataframes, each dataframe for one method

        # run the simulation for each method
        for method in methods:

            lr_skews, lr_means, lr_stds = self.simulate(method=method)

            # set the dummy variable indicators
            indicators = {col: np.zeros(self.l) for col in methods}
            indicators[method] = np.ones(self.l)

            # put the results in a dataframe 
            df = pd.DataFrame({
                "longrun_skews": lr_skews,
                "longrun_means": lr_means,
                "longrun_stds": lr_stds,
                'constant': np.ones(self.l),
                'sizes': self.sizes,
                'horizons': self.horizons,
                **indicators
            })

            # append to list
            dfs.append(df)

        # concatenate the dataframes in the list
        df_final = pd.concat(dfs, axis=0, ignore_index=True)

        return df_final


    def run_regression(self, data_path=None):
        """
        Arguments:
        - data_path: if available, provide the path to regression data

        Returns: Nothing

        """

        # if there is no path specified, then we must generate the dataset
        if data_path == None:
            reg_data = self.get_reg_data()
            reg_data.to_csv(f"./REG_DATA/reg_data_sector{int(self.sector)}_{self.n_outcomes}.csv") # save the data for later use
        # simply read the data if a path is given
        else:
            reg_data = pd.read_csv(data_path, header=0, index_col=0)

        # run the regressions
        self.regression(reg_data, dep_var="skew")
        self.regression(reg_data, dep_var="ratio")



