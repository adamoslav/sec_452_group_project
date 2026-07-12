# class to calculate portfolio and monetary value of the system and one individual agent
from aggregation_agent import aggregation_agent
import yfinance as yf
import pandas as pd

class portfolio_money:

    # class constructor
    def __init__(self, company, simulation_year, starting_cash, agent, low_weight_agent):
        self._company = company
        self._simulation_year = simulation_year
        self._starting_cash = starting_cash
        self._agent = agent
        self._low_weight_agent = low_weight_agent

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

    # calculating the "portfolio" value for the system
    def system_portfolio(self):

        # collecting yfinance stock price data
        data = self.collect_stock_price()

        # initialize aggregator agent
        aggregator_agent = aggregation_agent(self._company, self._simulation_year)

        # getting dataframes of decisions from aggregator agent
        df_decisions = aggregator_agent.get_decision(self._low_weight_agent)

        # initializing starting cash
        cash_available = self._starting_cash

        # collecting prices and storing it as a list
        prices = data["Close"].squeeze().to_list()

        # collecting decisions and storing it as list
        decisions = df_decisions["decision"].to_list()

        # initializing variable for shares
        shares = 0

        # initializing empty list for total portfolio + cash value
        total_value = []

        # looping over decisions and determining total value
        for i in range(len(decisions)):

            # setting up current day's price
            price = prices[i]

            # setting up decision for current day
            decision = decisions[i]

            # BUYING -> if decision is "BUY" and have enough cash
            if decision == "BUY" and cash_available > 0:

                # determining how many shares can be bought
                shares_to_buy = int(cash_available // price)

                # if can afford one or more shares -> calculate cost, update number of shares and update cash available
                if shares_to_buy >= 1:

                    # calculating cost of shares to be bought
                    cost = shares_to_buy * price

                    # updating shares
                    shares += shares_to_buy

                    # updating cash
                    cash_available -= cost

            # SELLING -> if decision is "SELL" and have more than 1 share
            elif decision == "SELL" and shares >= 1:

                # calculating revenue for all shares to be sold
                revenue = shares * price

                # updating cash_available -> increasing by revenue gained by selling all shares
                cash_available += revenue

                # updating shares -> selling all shares
                shares = 0


            # HOLD -> nothing needs to be updated

            # updating total_value by cash_available and value of stocks currently having
            total_value.append(cash_available + (shares * price))

        # copying the dataframe of decision to a new dataframe
        df_system = df_decisions.copy()

        # adding the values from total_value list to the dataframe
        df_system["system_dollar_value"] = total_value
        # returning the dataframe
        return df_system
    
    #########################################

    # calculating the "portfolio" value for individual agent and adding it to the dataframe for the system
    def agent_portfolio(self):

        # collecting the dataframe from system_portfolio function
        df_system = self.system_portfolio()

        ## checking which agent is passed to the function:

        # technical agent
        if self._agent.lower() == "technical":

            # collecting decisions and storing it as list
            decisions = df_system["technical_pred"].to_list()


        # health agent
        elif self._agent.lower() == "health":

            # collecting decisions and storing it as list
            decisions = df_system["health_pred"].to_list()

        # else sentiment agent
        else:

            # collecting decisions and storing it as list
            decisions = df_system["semantic_pred"].to_list()
        
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

        # looping over decisions and determining total value
        for i in range(len(decisions)):

            # setting up current day's price
            price = prices[i]

            # setting up decision for current day
            decision = decisions[i]

            # BUYING -> if decision is "BUY" (1) and have enough cash
            if decision == 1 and cash_available > 0:

                # determining how many shares can be bought
                shares_to_buy = int(cash_available // price)

                # if can afford one or more shares -> calculate cost, update number of shares and update cash available
                if shares_to_buy >= 1:

                    # calculating cost of shares to be bought
                    cost = shares_to_buy * price

                    # updating shares
                    shares += shares_to_buy

                    # updating cash
                    cash_available -= cost

            # SELLING -> if decision is "SELL" (-1) and have more than 1 share
            elif decision == -1 and shares >= 1:

                # calculating revenue for all shares to be sold
                revenue = shares * price

                # updating cash_available -> increasing by revenue gained by selling all shares
                cash_available += revenue

                # updating shares -> selling all shares
                shares = 0


            # HOLD (0) -> nothing needs to be updated

            # updating total_value by cash_available and value of stocks currently having
            total_value.append(cash_available + (shares * price))

        # adding the agent's total portfolio value to the dataframe
        df_system["agent_dollar_value"] = total_value
        
        # returning the dataframe
        return df_system
















    