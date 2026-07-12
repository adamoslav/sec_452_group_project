## creating a class for data preparation:

# importing libraries
import pandas as pd
import yfinance as yf
import requests
import time

# class 
class data_preparation:

    # class constructor
    def __init__(self, company, simulation_year):
        self._company = company
        self._simulation_year = simulation_year

    #########################################

    ## ACTIONS

    # 1. function for actions (target variable)
    def calculate_actions(self):

        # getting historical data
        historical_data = yf.download(
            self._company,
            start=f"{self._simulation_year - 2}-01-01",
            end=f"{self._simulation_year - 1}-12-31"
        )

        # setting uniform index
        historical_data.index = historical_data.index.tz_localize(None)

        # creating list of closing prices
        price_close = historical_data["Close"].values.flatten().tolist()

        # creating an empty list for actions
        actions = []

        # 1% percentage change threshold
        threshold = 0.01

        # looping over prices and determining action - 0 hold, 1 buy, -1 sell
        for i in range(len(price_close)):
            
            # check whether the current day can look 5 days ahead from current day and calculate percentage change
            if len(price_close) - i > 5:

                # calculating percentage change
                pct_change = ((price_close[i + 5] - price_close[i]) / price_close[i])

            # when last 5 days are reached - look at 4 days ahead, 3 days, 2 days, 1 day, and on last day do the same as day before
            elif len(price_close) - i == 5:
                
                # calculating percentage change - ((new value - old value) / old value)
                pct_change = ((price_close[i + 4] - price_close[i]) / price_close[i])
            
            # 3 days ahead
            elif len(price_close) - i == 4:
                
                # calculating percentage change 
                pct_change = ((price_close[i + 3] - price_close[i]) / price_close[i])
            
            # 2 days ahead
            elif len(price_close) - i == 3:

                # calculating percentage change 
                pct_change = ((price_close[i + 2] - price_close[i]) / price_close[i])
            
            # 1 day ahead
            elif len(price_close) - i == 2:
                
                # calculating percentage change 
                pct_change = ((price_close[i + 1] - price_close[i]) / price_close[i])
            
            # previous day's action (last day)
            else:
                
                # appending previous day's action
                actions.append(actions[-1])

                continue

            # if pct change is greater or equal than 0.01 (threshold) - buy, if less than or equal to -0.01, else hold
            if pct_change >= threshold:
                actions.append(1)
            elif pct_change <= (-threshold):
                actions.append(-1)
            else:
                actions.append(0)
            
        # returning actions
        return pd.Series(actions, index=historical_data.index)
    
    #########################################

    ## TECHNICAL AGENT

    # 1. function for technical agent's training data
    def train_data_technical_agent(self):

        # getting historical data for daily price change, volume change, and price fluctuation
        historical_data = yf.download(
            self._company,
            start=f"{self._simulation_year - 2}-01-01",
            end=f"{self._simulation_year - 1}-12-31",
            progress=False
        )

        # getting data for volatility
        volatility_data = yf.download(
            "^VIX",
            start=f"{self._simulation_year - 2}-01-01",
            end=f"{self._simulation_year - 1}-12-31",
            progress=False
        )
        
        # since both datframes have different index (different timezones) - setting it to the same index
        historical_data.index = historical_data.index.tz_localize(None)
        volatility_data.index = volatility_data.index.tz_localize(None)

        # creating suffixes for volatility data
        volatility_data = volatility_data.add_suffix("_vix")

        # sorting indexes for both dataframes before the merge
        historical_data = historical_data.sort_index()
        volatility_data = volatility_data.sort_index()

        # the volatility may be reported on days which stock market is not open - joining the two dataframes by index (by finding the closest day
        # to the trading day)
        historical_data = pd.merge_asof(
            historical_data,
            volatility_data,
            left_index=True,
            right_index=True,
            direction="backward"
        )

        # creating a dataframe for technical agent
        df_technical = pd.DataFrame(index=historical_data.index)


        ## moving average (10-day):

        # 10-moving average - creates nulls for days 0-9
        df_technical["moving_average_10_day"] = historical_data["Close"].rolling(window=10).mean()

        # handling nulls for moving average in days 0-9
        df_technical["moving_average_10_day"] = df_technical["moving_average_10_day"].bfill()

        
        ## price:

        # daily price change
        df_technical["daily_price_change"] = historical_data["Open"] - historical_data["Close"]

        # dialy price fluctuation
        df_technical["daily_price_fluctuation"] = historical_data["High"] - historical_data["Low"]

        
        ## volatility:
        
        # volatility
        df_technical["volatility"] = historical_data["Close_vix"]

        # volatility change
        df_technical["volatility_change"] = historical_data["High_vix"] - historical_data["Low_vix"]


        ## volume change:

        # creating a list of volume values
        volumes = historical_data["Volume"].values.flatten().tolist()
        
        # volume change
        df_technical["volume_change"] = [volumes[i] - volumes[i - 1] if i != 0 else 0 for i in range(len(volumes))]


        ## action:

        # calling the function to calculate actions and creating a column for the target variable
        df_technical["action"] = self.calculate_actions()

        # returning the training data dataframe
        return df_technical
    

    # 2. function for technical agent's simulation data
    def simulation_data_technical_agent(self):

        # getting historical data for daily price change, volume change, and price fluctuation
        historical_data = yf.download(
            self._company,
            start=f"{self._simulation_year}-01-01",
            end=f"{self._simulation_year}-12-31",
            progress=False
        )

        # getting data for volatility
        volatility_data = yf.download(
            "^VIX",
            start=f"{self._simulation_year}-01-01",
            end=f"{self._simulation_year}-12-31",
            progress=False
        )
        
        # since both datfarames have different index (different timezones) - setting it to the same index and resetting the index
        historical_data.index = historical_data.index.tz_localize(None)
        volatility_data.index = volatility_data.index.tz_localize(None)

        # creating suffixes for volatility data
        volatility_data = volatility_data.add_suffix("_vix")

        # sorting indexes for both dataframes before the merge
        historical_data = historical_data.sort_index()
        volatility_data = volatility_data.sort_index()

        # the volatility may be reported on days which stock market is not open - joining the two dataframes by index (by finding the closest day
        # to the trading day)
        historical_data = pd.merge_asof(
            historical_data,
            volatility_data,
            left_index=True,
            right_index=True,
            direction="backward"
        )

        # creating a dataframe for technical agent
        df_technical = pd.DataFrame(index=historical_data.index)

        ## moving average (10-day):

        # 10-moving average - creates nulls for days 0-9
        df_technical["moving_average_10_day"] = historical_data["Close"].rolling(window=10).mean()

        # handling nulls for moving average in days 0-9
        df_technical["moving_average_10_day"] = df_technical["moving_average_10_day"].bfill()

        
        ## price:

        # daily price change
        df_technical["daily_price_change"] = historical_data["Open"] - historical_data["Close"]

        # dialy price fluctuation
        df_technical["daily_price_fluctuation"] = historical_data["High"] - historical_data["Low"]

        
        ## volatility:
        
        # volatility
        df_technical["volatility"] = historical_data["Close_vix"]

        # volatility change
        df_technical["volatility_change"] = historical_data["High_vix"] - historical_data["Low_vix"]


        ## volume change:

        # creating a list of volume values
        volumes = historical_data["Volume"].values.flatten().tolist()
        
        # volume change
        df_technical["volume_change"] = [volumes[i] - volumes[i - 1] if i != 0 else 0 for i in range(len(volumes))]

        # return simulation dataframe
        return df_technical


    #########################################

    ## HEALTH AGENT

    # 1. function for retreiving API data and putting it to the right format
    def get_health_agent_data(self):

        # setting the API KEY
        API = "G6OTTEU16IXL124Y" # or K34H6QMPMV2QNSC0

        # setting the company
        COMPANY = self._company

        # 1. retrieving income statements
        url_income_statements= f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={COMPANY}&apikey={API}"
        r_income_statements = requests.get(url_income_statements)
        income_statements = r_income_statements.json()
        time.sleep(20) # to avoid crash because of the API call limit per minute

        # 2. retrieving balance sheets
        url_balance_sheets = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={COMPANY}&apikey={API}"
        r_balance_sheets = requests.get(url_balance_sheets)
        balance_sheets = r_balance_sheets.json()
        time.sleep(20) # to avoid crash because of the API call limit per minute
        
        # 3. retreiving EPS (earnings-per-share)
        url_eps = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={COMPANY}&apikey={API}"
        r_eps = requests.get(url_eps)
        eps = r_eps.json()
        time.sleep(20) # to avoid crash because of the API call limit per minute

        # 4. retreiving stock's price
        stock_price = yf.download(
            COMPANY,
            start=f"{self._simulation_year - 3}-01-01",
            end=f"{self._simulation_year}-12-31",
            progress=False
        )   

        # extracting the quaterly reports from income statements, balance sheets, EPS, and converting it to invidual dataframes
        df_income_statements = pd.DataFrame(income_statements["quarterlyReports"])
        df_balance_sheets = pd.DataFrame(balance_sheets["quarterlyReports"])
        df_eps = pd.DataFrame(eps["quarterlyEarnings"])


        # extracting stock price and converting to a dataframe stock price, 
        df_price = pd.DataFrame(stock_price["Close"])

        ## setting indexes of all dataframes to datetime object and universal time:

        # income statements
        df_income_statements["fiscalDateEnding"] = pd.to_datetime(df_income_statements["fiscalDateEnding"])
        df_income_statements.set_index("fiscalDateEnding", inplace=True)
        df_income_statements.index = df_income_statements.index.tz_localize(None)

        # balance sheets
        df_balance_sheets["fiscalDateEnding"] = pd.to_datetime(df_balance_sheets["fiscalDateEnding"])
        df_balance_sheets.set_index("fiscalDateEnding", inplace=True)
        df_balance_sheets.index = df_balance_sheets.index.tz_localize(None)

        # EPS
        df_eps["fiscalDateEnding"] = pd.to_datetime(df_eps["fiscalDateEnding"])
        df_eps.set_index("fiscalDateEnding", inplace=True)
        df_eps.index = df_eps.index.tz_localize(None)

        # stock price
        df_price.index = df_price.index.tz_localize(None)

        
        ## selecting columns which to include in each dataframe
        df_income_statements = df_income_statements[["totalRevenue", "netIncome"]]
        df_balance_sheets = df_balance_sheets[["totalLiabilities", "totalShareholderEquity", "totalCurrentAssets", "totalCurrentLiabilities"]]
        df_eps = df_eps[["reportedEPS", "estimatedEPS"]]

        # sorting indexes of all dataframes for the merge
        df_price = df_price.sort_index()
        df_income_statements = df_income_statements.sort_index()
        df_balance_sheets = df_balance_sheets.sort_index()
        df_eps = df_eps.sort_index()

        # joining quaterly reports to one dataframe - if any dates differ, outer join ensures nothing is lost 
        df_quaterly_reports = df_income_statements.join([df_balance_sheets, df_eps], how="outer")

        # merging the stock price data and the quaterly reports data
        health_data = pd.merge_asof(
            df_price,
            df_quaterly_reports,
            left_index=True,
            right_index=True,
            direction="backward"
        )

        # renaming the column for stock price (currently is named afer the company's ticker)
        health_data.rename(columns={COMPANY: "Close"}, inplace=True)

        # the dataframe has 4 years of data now -> the quaterly reports data is joined to stock's price index (trading days) based on the quaterly
        # reports date, but these often are not the same (when the quaterly reports are filed, the stock market is often not open) -> that is why
        # I used the ".merge_asof()" method to join the quaterly reports to the closest day before the quaterly reports were filed and the stock market
        # was still open
        health_data = health_data.ffill()

        # converting all the columns to numbers -> the API returns values as strings(objects)
        health_data = health_data.apply(pd.to_numeric, errors="coerce")

        return health_data
    

    # 2. function for training data for health_agent
    def train_data_health_agent(self, **kwargs):

        # "kwargs" ensure there is no need to define training years when calling the function
        if kwargs:
            training_year_1 = self._simulation_year
            training_year_2 = self._simulation_year
        else:
            training_year_1 = self._simulation_year - 2
            training_year_2 = self._simulation_year - 1

        # setting the company and simulation year constants
        COMPANY = self._company
        SIMULATION_YEAR = self._simulation_year

        # calling the function to retrieve data for the health agent and storing the dataframe
        health_data = self.get_health_agent_data()

        # downloading same data technical agent uses - for indexing purposes
        technical_agent_data = yf.download(
            self._company,
            start=f"{training_year_1}-01-01",
            end=f"{training_year_2}-12-31",
            progress=False
        )

        # setting uniform index
        technical_agent_data.index = technical_agent_data.index.tz_localize(None)

        # creating empty dataframe for the final training dataframe and using technical agent's data for indexing to ensure same length
        df_health_agent = pd.DataFrame(index=technical_agent_data.index)

        ## creating features:

        # 1. revenue growth
        
        # extracting quarter revenues
        quarter_revenues = health_data["totalRevenue"].drop_duplicates()

        # calculating the revenue growth
        revenue_growth = quarter_revenues.pct_change()

        # creating a column for revenue growth and using the "revenue_growth" values
        df_health_agent["revenue_growth"] = revenue_growth

        # the column has revenue_growth for all quarters except the very first quarter of the dataset(the year prior to training years) - fill with 0, else use forward fill
        df_health_agent["revenue_growth"] = df_health_agent["revenue_growth"].ffill().fillna(0)

        # 2. net income
        df_health_agent["net_income"] = health_data["netIncome"]

        # 3. profit margin
        df_health_agent["profit_margin"] = health_data["netIncome"] / health_data["totalRevenue"]

        # 4. debt-to-equity ration
        df_health_agent["debt_to_equity_ratio"] = health_data["totalLiabilities"] / health_data["totalShareholderEquity"]

        # 5. current ration
        df_health_agent["current_ratio"] = health_data["totalCurrentAssets"] / health_data["totalCurrentLiabilities"]

        # 6. earnings per share (EPS)
        df_health_agent["EPS"] = health_data["reportedEPS"]

        # 7. price-to-earnings ratio
        df_health_agent["price_to_earnings_ratio"] = health_data["Close"] / (health_data["reportedEPS"] * 4)

        # 8. action - if kwargs are provided, the function creates simulation dataset and actions are not needed
        if kwargs:
            
            # returning the dataframe without actions - simulation
            return df_health_agent.loc[str(training_year_1):str(training_year_2)]
        
        # if kwargs are not provided, the function creates training dataset and actions are provided
        else:
            
            # adding an action column
            df_health_agent["action"] = self.calculate_actions()

            # returning final dataframe with years for training
            return df_health_agent.loc[str(training_year_1):str(training_year_2)]

    

    # 3. function for simulation data for health_agent
    def simulation_data_health_agent(self):

        # setting the company and simulation year constants
        COMPANY = self._company
        SIMULATION_YEAR = self._simulation_year

        # calling the train_data_health_agent function
        df_health_agent = self.train_data_health_agent(simulation=True)

        # returning the final dataframe for simulation
        return df_health_agent







