Introduction :

    This program can calculate VaR and CVaR with a few main methods.
    VaR and CVaR are estimates that quantify that amount of risk held in a portfolio of stocks.

How to Use my Program :

    Reccomended to use PyCharm Community edition with Python 3.9 (What I used to develop / run the code):

        - Use Git to checkout the project file
        - Download the dependencies in Requirements.txt if not automatic.
        - Open var.py inside of risk directory
        - And call the classes and methods accordingly, some commented out examples.

        - Or to simply use the GUI to estimate a risk for a stock + option portfolio
            Run the gui.py file.


Features :

    Calculate estimate for :

        Historical Simulation  : VaR, CVaR
        Parametric Normal      : VaR, CVaR
        Parametric t-dist      : VaR, CVaR
        Parametric EWMA        : VaR
        Monte Carlo Simulation : VaR, CVaR
        Monte Carlo (options)  : VaR

    Backtests :
        The following backtesting methods work for all VaR estimates above.

            Kupiec POF test
            Standard Coverage test

    Plot Estimates for :

        Monte Carlo Simulation

Implementation & file structure :

    At the moment, all of my (C)VaR estimates are in one Python class, VaR which is inside the risk directory.

        This class contains an __init__ method which initialises each VaR object with the data from Yahoo Finance API.

        The rest of the methods are just used for computing (C)VaR.

        Then we create VaR Objects and print / plot the results.
