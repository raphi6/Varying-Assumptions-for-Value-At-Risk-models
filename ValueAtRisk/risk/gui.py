import PySimpleGUI as sg
import datetime as datetime
import numpy as np
import var


alpha = ""
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=270)
initial_investment = 0
timeframe = 0
stocks = []
weights = np.array([])
option_on = False

layout = [
    [sg.Text("Welcome to the Value at Risk application")],

    [sg.Text("Enter your confidence level: "), sg.InputText()],
    [sg.Text("Enter your investment in Dollars "), sg.InputText()],
    [sg.Text("Enter your time frame to calculate VaR over "), sg.InputText()],
    [sg.Text("Enter a list of stocks separated by commas(e.g AAPL, TSLA):")],
    [sg.Multiline(key='-STOCKS-')],
    [sg.Text("Enter the corresponding weights separated by commas(e.g 0.30, 0.70):")],
    [sg.Multiline(key='-WEIGHTS-')],

    [sg.Text("                         ")],
    [sg.Text("Fill out the following if you would like to add a European put or call option")],
    [sg.Text("Enter 'True'"), sg.InputText()],
    [sg.Text("Enter the stock your option is on, it has to already be in portfolio above"), sg.InputText()],
    [sg.Text("Enter the strike price"), sg.InputText()],
    [sg.Text("Enter your interest rate"), sg.InputText()],
    [sg.Text("Enter your option type, 'put' or 'call'"), sg.InputText()],
    [sg.Text("Enter your time to maturity as a decimal, eg 36.5 days: 0.1"), sg.InputText()],
    [sg.Text("Enter the number of options you have"), sg.InputText()],


    [sg.Button("Submit")],
    [sg.Button("End")]

]



window = sg.Window("Your Parameters", layout)

while True:
    event, values = window.read()

    if event in (None, "Submit"):
        alpha = 100 - int(values[0])



        initial_investment = int(values[1])

        timeframe = int(values[2])

        stocks = values['-STOCKS-'].split(',')
        stocks = [s.strip() for s in stocks]  # Remove whitespace from each symbol

        weights = values['-WEIGHTS-'].split(',')
        weights = [float(w.strip()) for w in weights]  # Remove whitespace from each weight
        weights = np.array(weights)
        weights /= np.sum(weights)  # Normalize the weights

        option_on = bool(values[3])

        if option_on:
            option_stock = values[4]
            strike_price = float(values[5])
            interest_rate = float(values[6])
            put_call = values[7]
            time_maturity = float(values[8])
            num_of_options = int(values[9])

        break

    if event == "End" or event == sg.WIN_CLOSED:
        break

    window.close()



if option_on == True:

    mcVaR = var.VaR(stocks, start_date, end_date, weights, alpha).monte_carlo_var(timeframe, initial_investment,
                                                                                              option_on, option_stock,
                                                                                              strike_price,
                                                                                              interest_rate,
                                                                                              put_call, time_maturity,
                                                                                              num_of_options)

    mcVaRrounded = round(mcVaR, 2)
    mcVaRstring = str(mcVaRrounded)

    mcVaRfinal = "Your Value at Risk is: " + mcVaRstring

    finalOptions = "Your Value at Risk is: " + mcVaRstring
    finalStocks = "Your Value at Risk is: "

else:


    mcVaR = var.VaR(stocks, start_date, end_date, weights, alpha).monte_carlo_var(timeframe, initial_investment)
    mcCVaR = var.VaR(stocks, start_date, end_date, weights, alpha).monte_carlo_cvar(timeframe, initial_investment)


    mcVaRrounded = round(mcVaR, 2)
    mcVaRstring = str(mcVaRrounded)

    mcVaRfinal = "Your Value at Risk is: " + mcVaRstring

    mcCVaRrounded = round(mcCVaR, 2)
    mcCVaRstring = str(mcCVaRrounded)

    finalStocks = "Your Value at Risk is: " + mcVaRstring + " and your Conditional Value at Risk is: " + mcCVaRstring

    finalOptions = "Your Value at Risk is: " + mcVaRstring






layoutstocks = [
    [sg.Text("Your Value at Risk estimates: ")],

    [sg.Text(finalStocks)],

    [sg.Button("End")]
]

layoutoptions = [
    [sg.Text("Your Value at Risk estimate: ")],

    [sg.Text(finalOptions)],

    [sg.Button("End")]

]

if option_on == True:
    window2 = sg.Window("Your Value at Risk", layoutoptions)
else:
    window2 = sg.Window("Your Parameters", layoutstocks)



while True:
    event, values = window2.read()

    if event == "End" or event == sg.WIN_CLOSED:
        break

    window2.close()





