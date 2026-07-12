# this class serves as a baseline approach for simulation -> buy and hold for the entire simulation
import pandas as pd
import yfinance as yf

class baseline_approach:

    # class constructor
    def __init__(self, company, simulation_year, starting_cash):
        self._company = company
        self._simulation_year = simulation_year
        self._starting_cash = starting_cash

    #########################################

    # collect yfinance data - need to extract price
    def collect_stock_price(self):

        # downloading the yfinance data -> stock's price
        price_data = yf.download(
            self._company,
            start=f"{self._simulation_year}-01-01",
            end=f"{self._simulation_year}-12-31",
            progress=False
        )

        # returning stock price data
        return price_data
    
    #########################################

    # calculating "portfolio" value for baseline approach
    def baseline_portfolio(self):

        # collecting yfinance stock price data
        data = self.collect_stock_price()

        # initializing starting cash
        cash_available = self._starting_cash

         # collecting prices and storing it as a list
        prices = data["Close"].squeeze().to_list()

        # initializing variable for shares
        shares = 0

        # initializing empty list for total portfolio + cash value
        total_value = []

        # looping over each day in simulation and calculating total_value
        for i in range(len(prices)):

            # getting price
            price = prices[i]

            # first day of the simulation -> BUY, update cash and shares
            if i == 0:

                # determining how many shares can be bought
                shares_to_buy = int(cash_available // price)

                # calculating cost of shares to be bought
                cost = shares_to_buy * price

                # updating shares
                shares += shares_to_buy

                # updating cash
                cash_available -= cost

               # updating total_value by cash_available and value of stocks that were bought
                total_value.append(cash_available + (shares * price))

            # else recalculating total_value by cash_available and value of stocks that were bought
            else:
                total_value.append(cash_available + (shares * price))

         # creating dataframe from total_value list
        df_baseline = pd.DataFrame(total_value, columns=["baseline_dollar_value"], index=data.index)

        # returning the dataframe
        return df_baseline
        
