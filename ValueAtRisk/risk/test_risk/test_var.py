from risk import var
# import pytest
import datetime as datet
import numpy as np
from risk.var import VaR

"""
The get_data function has been replaced by VaR Class __init__ method.


def test_get_data():
    end_date = datet.datetime(2022, 2, 2)
    start_date = end_date - datet.timedelta(days=100)
    stock_list = ['AAPL', 'TSLA', 'GOOGL', 'RTX']

    # should show 70 days of market data
    row_num = len(var.get_data(stock_list, start_date, end_date)[0])
    assert row_num == 70
    
"""


def test_var_init():
    """

    test_var_init is a new unit test created that replaces the test for get_data which was replaced by __init__
     in VaRClass.
     VaR instance is checked to see if the data has the right number of rows.
    """
    end_date = datet.datetime(2022, 2, 2)
    start_date = end_date - datet.timedelta(days=100)
    stock_list = ['AAPL', 'TSLA', 'GOOGL', 'RTX']
    weights = np.array([0.20, 0.50, 0.20, 0.10])
    alpha = 5

    row_num = len(VaR(stock_list, start_date, end_date, weights, alpha).get_daily_return())
    assert row_num == 69  # Changed from 70 to 69, new API yfinance


def test_historical_cvar():
    """

    Will check if Expected Shortfall (hcvar) is a bigger loss than Historical VaR (hvar), at a pre-determined time-
     frame, confidence level (alpha) and weight.

     Expected Shortfall should usually always be a bigger loss/risk than VaR, with given stock data it will always
     be true.

    """
    end_date = datet.datetime(2022, 2, 2)
    start_date = end_date - datet.timedelta(days=400)
    stock_list = ['AAPL', 'TSLA', 'GOOGL', 'RTX']
    weights = np.array([0.20, 0.50, 0.20, 0.10])
    alpha = 5

    t = 20  # time frame to calculate VaR over.

    hvar = VaR(stock_list, start_date, end_date, weights, alpha).historical_var()*np.sqrt(t)
    hcvar = VaR(stock_list, start_date, end_date, weights, alpha).historical_cvar()*np.sqrt(t)

    assert (hcvar < hvar)

def test_conditional_cvar():
    """

    Will check if Expected Shortfall (pcvar) is a bigger loss than Parametric VaR (pvar), at a pre-determined time-
     frame, confidence level (alpha) and weight.

     Expected Shortfall should usually always be a bigger loss/risk than VaR, with given stock data it will always
     be true.

    """
    end_date = datet.datetime(2022, 2, 2)
    start_date = end_date - datet.timedelta(days=400)
    stock_list = ['AAPL', 'TSLA', 'GOOGL', 'RTX']
    weights = np.array([0.20, 0.50, 0.20, 0.10])
    alpha = 5

    t = 20  # time frame to calculate VaR over.

    pvar = -VaR(stock_list, start_date, end_date, weights, alpha).parametric_var(t)
    pcvar = -VaR(stock_list, start_date, end_date, weights, alpha).parametric_cvar(t)

    assert (pcvar < pvar)

def test_montecarlo_cvar():
    """

    Will check if Expected Shortfall (pcvar) is a bigger loss than Monte Carlo  VaR , at a pre-determined time-
     frame, confidence level (alpha) and weight.

     Expected Shortfall should usually always be a bigger loss/risk than VaR, with given stock data it will always
     be true.

    """
    end_date = datet.datetime(2022, 2, 2)
    start_date = end_date - datet.timedelta(days=400)
    stock_list = ['AAPL', 'TSLA', 'GOOGL', 'RTX']
    weights = np.array([0.20, 0.50, 0.20, 0.10])
    alpha = 5

    t = 20  # time frame to calculate VaR over.

    monteCarloVaR = VaR(stock_list, start_date, end_date, weights, alpha).monte_carlo_var()
    monteCarloCVaR = VaR(stock_list, start_date, end_date, weights, alpha).monte_carlo_cvar()

    assert ( monteCarloVaR < monteCarloCVaR)