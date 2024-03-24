import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
import datetime as datet
from pandas_datareader import data as pandasdr
from scipy.stats import norm, chi2
from scipy.stats import t as tdist
import yfinance as yf


class OptionPricing:

    def black_scholes(S, X, T, r, v, option='call'):
        """

        Computes the price of a European put or call option using the Black-Scholes formula.

        :param X: strike price
        :param T: time to maturity
        :param r: interest rate
        :param v: volatility
        :param option: 'put' or 'call'
        :return: the price of an option
        """
        d1 = (math.log(S / X) + (r + 0.5 * v ** 2) * T) / (v * math.sqrt(T))
        d2 = (math.log(S / X) + (r - 0.5 * v ** 2) * T) / (v * math.sqrt(T))

        if option == 'call':
            option_price = S * norm.cdf(d1) - X * math.exp(-r * T) * norm.cdf(d2)

        elif option == 'put':
            option_price = X * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return option_price


class VaR:
    """
    For my __init__, portfolio_analysis, historical, parametric and monte carlo methods; I based my work from reference
    number 9 in my report:
    Author: Emerick, J.
    Date: 2020
    Available at: https://quantpy.com.au/risk-management/value-at-risk-var-and-conditional-var-cvar/
    The __init__ and portfolio_analysis methods are copied from him as this data collection can only be done one way in
    the pandas-datareader library. However, I have adapted the rest of his code in my own way and completely understand
    them, I even believe these algorithms come under common knowledge as it is just mathematical formulae that can be
    found in multiple textbooks.
    """

    def __init__(self, s, start, end, weights, alpha):
        """
        This initialization method builds each VaR object by firstly gathering Closing market price from Yahoo API. Then
        it computes fundamental parameters that are needed for simple VaR calculations

        :param s: list of stocks we want to gather data for.
        :param start: datetime object, gathering data from this date to the end date
        :param end: datetime object, the end date.
        :param weights: list of weights that show how much of each stock we own in portfolio.
        :param alpha: integer, (100 - confidence level)
        """
        yf.pdr_override()
        yahoo_data = pandasdr.get_data_yahoo(s, end=end, start=start)['Close']

        self.stock_prices = yahoo_data
        self.stock_prices_mean = self.stock_prices.mean()
        self.daily_returns = yahoo_data.pct_change()
        self.daily_returns.dropna()  # Getting rid of the most recent price, it is n/a as we cant compute return for it.
        self.covariance_matrix = self.daily_returns.cov()  # Computing covariance matrix
        self.mean_returns = self.daily_returns.mean()
        self.weights = weights
        self.alpha = alpha
        self.daily_returns['portfolio'] = self.daily_returns.dot(self.weights)  # New column that takes returns in terms
        # of weights.
        self.std_daily_returns = self.daily_returns.std()

    def portfolio_analysis(self, time):
        """
        Performing extra calculations that are needed for certain methods.

        time - Lookback period to analyse our portfolio over.

        :return: mean_weighted_return: which are the returns that have been taken the mean of with respect to weighting
                and time.

                standard deviation of the portfolio: this is the standard deviation of the portfolio using covariance
                matrix, weights and time from above.
        """

        mean_weighted_return = time * np.sum(self.weights * self.mean_returns)
        port_vol_right = np.dot(self.covariance_matrix, self.weights)
        port_vol_left = np.dot(self.weights.T, port_vol_right)

        return mean_weighted_return, (np.sqrt(t) * np.sqrt(port_vol_left))

    """ Getter methods """

    def get_daily_return(self):
        return self.daily_returns

    """  VaR Calculation methods  """

    def historical_var(self):
        """
        Computes alpha percentile of the historical return distribution.

        :return: The Value at Risk percentile point from historical simulation.
        """

        VaR = np.percentile(self.daily_returns['portfolio'].dropna(), self.alpha)

        return VaR

    def historical_cvar(self):
        """
        Computes the expectation/mean of the bottom percentile from the historical distribution.

        :return: The Conditional Value at Risk, expectation of lower percentile, from historical simulation.
        """
        bottom_percentile = []
        VaR = self.historical_var()
        for i in self.daily_returns['portfolio']:
            if VaR > i:
                bottom_percentile.append(i)

        CVaR = mean(bottom_percentile)

        return CVaR

    def parametric_var(self, time, distribution='gaussian'):
        """
        Computes alpha percentile of the Gaussian return distribution based of historical parameters.

        :param time: time horizon to calculate VaR for.
        :param distribution: distribution that we are using.
        :return: the VaR percentile point from Gaussian distribution.
        """
        mean_weighted_return, std_deviation = self.portfolio_analysis(time)

        if distribution == 'gaussian':
            conf_level = 1 - self.alpha / 100
            norm_point = norm.ppf(conf_level)

            VaR = norm.ppf(conf_level) * std_deviation

        return VaR - mean_weighted_return

    def parametric_cvar(self, time, distribution='gaussian'):
        """
        Computes the expectation/mean of the bottom percentile from the Gaussian distribution based of historical
        parameters.
        Equation used for CVaR is:

                inv(alpha) * SNDF( Phi(alpha) ) * standardDeviation - meanReturn

                where SNDF is standard normal density function and Phi gives us the alpha quantile of standard normal
                distribution.

        :param time: time horizon to calculate CVaR for.
        :param distribution: distribution that we are using.
        :return: The Conditional Value at Risk, expectation of lower percentile, from parametric approach.
        """
        mean_weighted_return, std_deviation = self.portfolio_analysis(time)

        inv_alpha = (alpha * 0.01) ** -1  # inverse alpha
        norm_point = norm.ppf(alpha * 0.01)
        VaR = inv_alpha * norm.pdf(norm_point) * std_deviation

        return VaR - mean_weighted_return

    def parametric_t_var(self, time, dof):
        """
        Computes alpha percentile of the Student's t-distribution of returns based of historical parameters.

        :param time: time horizon to calculate VaR over.
        :param dof: degrees of freedom
        :return: The Value at Risk
        """

        mean_weighted_return, std_deviation = self.portfolio_analysis(time)

        conf_level = 1 - self.alpha / 100
        t_point = tdist.ppf(conf_level, dof)
        VaR = np.sqrt((dof - 2) / dof) * t_point * std_deviation

        return VaR - mean_weighted_return

    def parametric_t_cvar(self, time, dof):
        """
        Computes the expectation/mean of the bottom percentile from the Gaussian distribution based of historical
        parameters.

        Made use of Equation 35 from Section 2.5.2 in report

        :param time: time horizon to calculate VaR over.
        :param dof: degrees of freedom
        :return: The Conditional Value at Risk
        """
        mean_weighted_return, std_deviation = self.portfolio_analysis(time)

        conf_level = self.alpha / 100
        v = tdist.ppf(conf_level, dof)
        CVaR = -1 / (self.alpha / 100) * (1 - dof) ** (-1) * (dof - 2 + v ** 2) * tdist.pdf(v, dof) * std_deviation

        return CVaR - mean_weighted_return

    def ewma_vol(self):
        """
        Calculating volatility using EWMA approach.

        In this method for EWMA we apply the weighted method where we calculate beforehand the weight each return shall
         have:


            Example:
                series = (100, 99, 98, ... 2, 1, 0)  #for 100 trading days
                weights = Lamda^(series) * 1 - (Lamda)     #array of 100 weights where most recent return has smallest
                                                           #series value, 0. And vice versa for oldest return.

                Equation:

                        EWMAVariance = Sum( weights * squaredReturns )

                Multiply weights by squared returns and the sum will  be EWMA variance and sqrt is volatility.

        As apposed to using the recursive formula, Equation 12, in Section 2.7.2

        :return: array of ewma volatilities
        """

        lamda = 0.94  # lamda value from JP Morgan RiskMetrics
        length = len(self.daily_returns)

        ewma_values = []
        std_values = []

        for i in self.daily_returns.columns:
            series = np.arange(length - 1, -1, -1)
            exp_array = []
            for j in series:
                exp = lamda ** j * (1 - lamda)
                exp_array.append(exp)
            product = np.power(self.daily_returns[i], 2)
            product = product * exp_array

            ewma_values.append(np.sqrt(product.sum()))

            std = self.daily_returns[i].std()
            std_values.append(std)

        return ewma_values

    def parametric_ewma_var(self, time):
        """
        Using the same method as we have for Gaussian parametric but instead using EWMA volatility.

        :param time: time horizon to calculate VaR for.
        :return: The Value at Risk.
        """
        ewma_volatility = self.ewma_vol()
        conf_level = 1 - self.alpha / 100
        var_values_dict = {}

        counter = 0

        for i in self.daily_returns.columns:
            column = i
            norm_point = norm.ppf(conf_level)
            VaR = ewma_volatility[counter] * np.sqrt(time) * norm_point

            var_values_dict.update({column: VaR})
            counter = counter + 1

        return var_values_dict

    def monte_carlo_portfolio(self, time=10, mc_simulations=10000, test_portfolio_val=100000, option_on=False,
                              option_on_stock='TSLA', strike_price=150, r=0.05, option_type='put',
                              time_to_maturity=30 / 365, num_options=1000):
        """
        Simulating multiple runs of the portfolio using Monte Carlo simulation with equation similar to Geometric
        Brownian Motion. Additionally simulates European put or call options into the portfolio.

        We will be following the formula to calculate future returns:

            Daily Returns t = MeanReturn + LZt

            where Zt are samples from normal distribution, L is a Cholesky decomposition of covariance matrix.

        With the addition of options, we simulate each underlying asset separetly as opposed to the above.

        :param mc_simulations: number of monte carlo simulations to be produced
        :param time: timeframe in days to simulate for.
        :return: portfolio_simulations: the results of our simulations, test_portfolio_val: Initial investment.
        """

        # Following variables for option portfolio

        if option_on:

            index = stock_list.index(option_on_stock)

            vol = self.std_daily_returns[option_on_stock]

            mean_index = self.mean_returns[index]

            last_stock_price = self.stock_prices[option_on_stock][-1]

            stock_simulations = np.zeros((mc_simulations, time))

            time_to_maturity_updated = time_to_maturity - time / 365  # Adjusting as simulation progresses.

        # Following variables for stock portfolio

        num_simulations = mc_simulations

        portfolio_simulations = np.full(shape=(time, num_simulations), fill_value=0.0)

        length_columns = len(self.daily_returns.columns) - 1
        fill = self.mean_returns
        mean_matrix = np.full(shape=(time, length_columns),
                              fill_value=fill)  # mean returns in terms of number of days

        cov_matrix = self.covariance_matrix

        length_returns_columns = len(self.daily_returns.columns)
        cholesky_lower_l = np.linalg.cholesky(cov_matrix)

        # Now entering the Monte Carlo simulation in the for loop.

        for i in range(0, mc_simulations):

            Z = np.random.normal(size=(time, length_returns_columns - 1))
            daily_returns = mean_matrix.T + np.inner(cholesky_lower_l, Z)
            weights_returns_dotproduct = np.inner(self.weights, daily_returns.T)
            portfolio_simulations[:, i] = test_portfolio_val * np.cumprod(weights_returns_dotproduct + 1)

            # If we have an option, simulate the underlying asset separately.
            if option_on:

                Z_option = Z[:, index]

                stock_prices = np.zeros(time)
                stock_prices[0] = last_stock_price

                for j in range(1, time):
                    stock_prices[j] = stock_prices[j - 1] * np.exp(
                        mean_index - (vol ** 2) / 2 + mean_index * Z_option[j])

                stock_simulations[i, :] = stock_prices

        # Now to value portfolio with algorithm in Section 2.8.2
        if option_on:

            # Valuing initial option price
            initial_s = stock_prices[0]
            option_price_initial = OptionPricing.black_scholes(initial_s, strike_price, time_to_maturity, r, vol
                                                               , option_type)

            test_portfolio_val = test_portfolio_val + num_options * option_price_initial

            # Valuing portfolio at the end of each simulation (the final prices)
            final_prices = stock_simulations[:, -1]

            temp_arr = []

            index = 0
            for s in final_prices:
                option_price = OptionPricing.black_scholes(s, strike_price, time_to_maturity_updated, r, vol
                                                           , option_type)
                temp_arr.append(option_price)

                portfolio_simulations[-1, index] = portfolio_simulations[-1, index] + num_options * option_price

                index = index + 1

            return portfolio_simulations, test_portfolio_val

        """
        plt.plot(portfolio_simulations)
        plt.ylabel('Portfolio value in dollars ')
        plt.xlabel('Number of days of simulation')
        plt.show()
        """
        return portfolio_simulations, test_portfolio_val

    def monte_carlo_var(self, t, initial_investment, option_on=False,
                              option_on_stock='TSLA', strike_price=190, r=0.05, option_type='put',
                              time_to_maturity=30 / 365, num_options=1000):
        """
        Computing VaR in dollar value in the same way we do for historical method, finding the alpha percentile.
        :return: The Value at Risk dollar value point from Monte Carlo simulation.
        """
        time = t
        mc_simulations = 20000
        test_portfolio_val = initial_investment

        port_sims, port_val = self.monte_carlo_portfolio(time, mc_simulations, test_portfolio_val)

        if option_on:

            port_sims, port_val = self.monte_carlo_portfolio(time, mc_simulations, test_portfolio_val, option_on,
                                                         option_on_stock, strike_price, r, option_type,
                                                         time_to_maturity, num_options)

        port_results = pd.Series(port_sims[-1, :])

        VaR = np.percentile(port_results, self.alpha)

        return port_val - VaR

    def monte_carlo_cvar(self, t, initial_investment):
        """
        Computing CVaR in dollar value in the same way we do for historical method, finding the bottom percentile set.
        :return: The Conditional Value at Risk, expectation/mean of bottom percentile, dollar value point from Monte
         Carlo simulation.
        """
        port_sims, port_val = self.monte_carlo_portfolio(time=t, mc_simulations=10000,
                                                         test_portfolio_val=initial_investment)
        test_portfolio_val = initial_investment

        port_results = pd.Series(port_sims[-1, :])

        var_point = self.monte_carlo_var(t, initial_investment)
        bottom_percentile = []

        # Looping through to find all values that are below VaR from Monte_carlo_var()
        for i in port_results:

            if i < test_portfolio_val:

                if test_portfolio_val - i >= var_point:
                    bottom_percentile.append(i)

        return test_portfolio_val - mean(bottom_percentile)


""" Below are parameters that we can choose to compute (C)VaR for """

"Initial Investment"
InitialInvestment = 10000

"Dates for collection of historical data"

end_date = datet.datetime.now()  # - datet.timedelta(days=1600)
start_date = end_date - datet.timedelta(days=270)

print(end_date)
print(start_date)

"Stocks in our portfolio"
# stock_list = ['HD']
# stock_list = ['GOOGL', 'MSFT', 'AAPL']

"Weights of the above stocks"
# weights = np.array([1.00])
# weights = np.array([0.33, 0.33, 0.34])

stock_list = ['AAPL', 'TSLA']

weights = np.array([0.01, 0.99])

"Confidence Level"
alpha = 10

"time frame to calculate VaR over."
t = 10

""" Creating VaR objects that we can find (C)VaR with. """

# hVaR = -VaR(stock_list, start_date, end_date, weights, alpha).historical_var() * np.sqrt(t)
# hCVaR = -VaR(stock_list, start_date, end_date, weights, alpha).historical_cvar() * np.sqrt(t)

# pVaR = VaR(stock_list, start_date, end_date, weights, alpha).parametric_var(t)
# pCVaR = VaR(stock_list, start_date, end_date, weights, alpha).parametric_cvar(t)

# p_t_VaR = VaR(stock_list, start_date, end_date, weights, alpha).parametric_t_var(t, 6)
# p_t_CVaR = VaR(stock_list, start_date, end_date, weights, alpha).parametric_t_cvar(t, 6)


# ewmaVaR = VaR(stock_list, start_date, end_date, weights, alpha).parametric_ewma_var(t)

# monteCarloVaR = VaR(stock_list, start_date, end_date, weights, alpha).monte_carlo_var(t, InitialInvestment)
# monteCarloCVaR = VaR(stock_list, start_date, end_date, weights, alpha).monte_carlo_cvar(t, InitialInvestment)

# portReturns, portStd = VaR(stock_list, start_date, end_date, weights, alpha).portfolio_analysis(t)

"""Printing Expected portfolio return and (C)VaR values"""


# print("Historical VaR incl. portfolio:          ", round(hVaR * InitialInvestment, 2))
# print("Historical CVaR:         ", round(hCVaR * InitialInvestment, 2))

# print("Normal VaR:              ", round(pVaR * InitialInvestment, 2))
# print("Normal CVaR:             ", round(pCVaR * InitialInvestment, 2))

# print("Student-t VaR:             ", round(p_t_VaR * InitialInvestment, 2))
# print("Student-t CVaR:             ", round(p_t_CVaR * InitialInvestment, 2))


# print("EWMA VaR:                ", round(InitialInvestment * ewmaVaR['portfolio'], 2))

# print("MonteCarlo VaR:          ", round(monteCarloVaR, 2))
# print("MonteCarlo CVaR:         ", round(monteCarloCVaR, 2))


class Backtest:
    """
        In this class we will look over a historical period of data and count the number of violations that occur with
        our VaR models. We will then perform standard and Kupiec coverage tests to check the reliability of the models.
        """
    def __init__(self):
        """
        Getting the historical data and setting it up for VaR calculations.
        """
        self.test_end_date = datet.datetime.now() - datet.timedelta(days=2200)
        self.test_start_date = self.test_end_date - datet.timedelta(days=1200)

        self.delta = self.test_end_date - self.test_start_date

        yf.pdr_override()
        self.yahoo_data = pandasdr.get_data_yahoo(stock_list, end=self.test_end_date, start=self.test_start_date)[
            'Close']
        self.yahoo_data = self.yahoo_data.pct_change()
        self.yahoo_data = self.yahoo_data.dropna()
        self.yahoo_data['portfolio'] = self.yahoo_data.dot(weights)
        self.yahoo_data.index = pd.to_datetime(self.yahoo_data.index)

    def violation_count(self):
        """
        For each VaR model, we will count the number of violations that occur.

        This is done by iterating over the number of backtest days, and for each day:

                We compute our VaR estimates with a 1-day time horizon

            Then compare this VaR to the next day in history. Counting a violation if that daily return is lower
            than our VaR.

        :return: dictionary containing the model and its number of violations.
        """
        test_start_date = self.test_start_date
        test_end_date = self.test_end_date
        delta = self.delta

        print(test_start_date)
        print(test_end_date)
        print(delta)

        num_of_days = 0
        violations_historical = 0
        violations_parametric = 0
        violations_parametric_t = 0
        violations_ewma = 0
        violations_monte_carlo = 0

        for i in range(delta.days + 1):
            day = test_start_date + datet.timedelta(days=i)

            day_str = day.strftime('%Y-%m-%d')
            if day_str in self.yahoo_data.index:  # using day as str obj here
                num_of_days = num_of_days + 1
                test_day = day  # using datetime object instead of str
                test_day_return = self.yahoo_data.loc[day_str]['portfolio']

                print(test_day, " : ", test_day_return)

                temp_end = test_day - datet.timedelta(days=1)  # offset, finding VaR for test_day
                temp_start = temp_end - datet.timedelta(days=270)

                varObj = VaR(stock_list, temp_start, temp_end, weights, alpha)

                var_historical = varObj.historical_var() * np.sqrt(t)
                var_parametric = -varObj.parametric_var(t)
                var_parametric_t = -varObj.parametric_t_var(t, 4)
                var_ewma_dict = varObj.parametric_ewma_var(t)
                var_ewma = -var_ewma_dict['portfolio']
                var_montecarlo = -varObj.monte_carlo_var(t, 1)

                print(self.yahoo_data.loc[day_str]['portfolio'], "Value at Risk Historical = :", var_historical)
                print(self.yahoo_data.loc[day_str]['portfolio'], "Value at Risk Parametric = :", var_parametric)
                print(self.yahoo_data.loc[day_str]['portfolio'], "Value at Risk Param. t-dist = :", var_parametric_t)
                print(self.yahoo_data.loc[day_str]['portfolio'], "Value at Risk EWMA = :", var_ewma)
                print(self.yahoo_data.loc[day_str]['portfolio'], "Value at Risk Monte Carlo = :", var_montecarlo)

                if var_historical > test_day_return:
                    violations_historical = violations_historical + 1

                if var_parametric > test_day_return:
                    violations_parametric = violations_parametric + 1

                if var_parametric_t > test_day_return:
                    violations_parametric_t = violations_parametric_t + 1

                if var_ewma > test_day_return:
                    violations_ewma = violations_ewma + 1

                if var_montecarlo > test_day_return:
                    violations_monte_carlo = violations_monte_carlo + 1

        print("Number of violations Historical: ", violations_historical)
        print("Number of violations Parametric: ", violations_parametric)
        print("Number of violations Parametric t-dist: ", violations_parametric_t)
        print("Number of violations EWMA: ", violations_ewma)
        print("Number of violations Monte Carlo: ", violations_monte_carlo)
        print("Number of days for Testing Period : ", num_of_days)

        violation_dict = {
            "historical": [violations_historical],
            "parametric": [violations_parametric],
            "parametric t-dist": [violations_parametric_t],
            "ewma": [violations_ewma],
            "monte_carlo": [violations_monte_carlo],
            "test_length": num_of_days
        }

        return violation_dict

    def proportion(self):
        """
        Calculating the proportion of violations compared to the total number of backtest days, we do this for
        each method

        :return: updated dictionary with proportions.
        """
        violation_dict = self.violation_count()

        test_dict = {
            "historical": [12],
            "parametric": [56],
            "parametric t-dist": [70],
            "ewma": [45],
            "monte_carlo": [55],
            "test_length": 826
        }
        test_length = test_dict["test_length"]

        test_dict = violation_dict
        test_length = violation_dict["test_length"]

        for key in list(test_dict.keys())[:5]:
            violations = test_dict[key][0]  # Get violations from dictionary
            proportion = violations / test_length  # proportion of violations to test_length (backtest days)
            test_dict[key].append(proportion)

        # print(test_dict)
        return test_dict

    def standard_coverage(self):
        """
        As described in Section 2.9.1, but instead computes p-values as opposed to an interval
        """
        proportion_dict = self.proportion()
        print(proportion_dict)

        for key in list(proportion_dict.keys())[:5]:
            proportion_violations = proportion_dict[key][1]

            theoretical_violations = alpha / 100

            sample_size = proportion_dict["test_length"]

            difference = proportion_violations - theoretical_violations

            standard_error = math.sqrt(theoretical_violations * (1 - theoretical_violations) / sample_size)

            z_stat = difference / standard_error

            p_value = (2 * (1 - abs(norm.sf(z_stat))))

            print("VaR method: ", key, "  z-stat: ", z_stat, "   p-value: ", p_value)

    def kupiec_coverage(self):
        """
        As described in Section 2.9.2, but instead computes p-values as opposed to an interval
        """
        proportion_dict = self.proportion()
        print(proportion_dict)

        for key in list(proportion_dict.keys())[:5]:
            proportion_violations = proportion_dict[key][1]  # violations per backtest size

            proportion_nonviolations = 1 - proportion_violations  # non violations per backtest size

            theoretical_nonviolations = (100 - alpha) / 100  # expected non violations per backtest size

            theoretical_violations = alpha / 100  # expected violations per backtest size

            freq_nonviolations = proportion_dict["test_length"] - proportion_dict[key][0]  # frequency of nonviolations

            freq_violations = proportion_dict[key][0]  # frequency of violations

            chi_squared = 2 * np.log(((proportion_nonviolations / theoretical_nonviolations) ** freq_nonviolations) * (
                    proportion_violations / theoretical_violations) ** freq_violations)

            p_value = chi2.sf(chi_squared, 1)  # p-value with DOF = 1

            print("VaR method: ", key, "  chi_squared: ", chi_squared, "   p-value: ", p_value)

""" Running our backtests """

# Backtest().kupiec_coverage()
# Backtest().standard_coverage()
