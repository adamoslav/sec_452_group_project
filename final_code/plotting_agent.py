# this class is for plotting agent to visualize the results of the simulation
from calculate_portfolio import portfolio_money
from baseline_approach import baseline_approach
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


class plotting_agent:

    # class constructor
    def __init__(self, company, simulation_year, starting_cash, agent, low_weight_agent):
        self._company = company
        self._simulation_year = simulation_year
        self._starting_cash = starting_cash
        self._agent = agent
        self._low_weight_agent = low_weight_agent

    #########################################

    # function to join dataframes - system and individual agent's portfolio, baseline "portfolio"
    def join_dataframes(self):

        # initialize calculate_portfolio instance
        system_and_agent = portfolio_money(self._company, self._simulation_year, self._starting_cash, self._agent, self._low_weight_agent)

        # initialize baseline_approach instance
        baseline = baseline_approach(self._company, self._simulation_year, self._starting_cash)

        # collecting dataframe for system
        # df_system = system_and_agent.system_portfolio()

        # collecting dataframe for system agent
        df_system_and_agent = system_and_agent.agent_portfolio()

        # collecting baseline dataframe
        df_baseline = baseline.baseline_portfolio()

        # adding the dataframes together
        df_master = pd.concat([df_system_and_agent, df_baseline], axis=1)

        # returning the dataframe
        return df_master
    
    #########################################

    # function to print returns (%), Sharpe Ratio, and Maximum Drawdown and create plots - comparison plot of portfolios over time, and agent's decision over time
    def plot_results(self):

        # collecting the master dataframe
        df_master = self.join_dataframes()

        # collecting data for risk-free-rate (10-year CBOE Interest Rate)
        cboe =  yf.download(
            "^TNX",
            start=f"{self._simulation_year}-01-01",
            end=f"{self._simulation_year}-12-31",
            progress=False
        )
        
        # calculating mean of the risk-free-rate for the simulation year
        risk_free_rate_mean = cboe["Close"].mean().item()

        # calculating portfolio return of all approaches - system, one agent, baseline approach
        system_return = (df_master["system_dollar_value"].iloc[-1] - self._starting_cash) / self._starting_cash
        agent_return = (df_master["agent_dollar_value"].iloc[-1] - self._starting_cash) / self._starting_cash
        baseline_return = (df_master["baseline_dollar_value"].iloc[-1] - self._starting_cash) / self._starting_cash

        # calculating maximum drawdown for all approaches - system, one agent, baseline approach
        system_drawdown = ((df_master["system_dollar_value"] - df_master["system_dollar_value"].cummax()) / df_master["system_dollar_value"].cummax()).min()
        agent_drawdown = ((df_master["agent_dollar_value"] - df_master["agent_dollar_value"].cummax()) / df_master["agent_dollar_value"].cummax()).min()
        baseline_drawdown = ((df_master["baseline_dollar_value"] - df_master["baseline_dollar_value"].cummax()) / df_master["baseline_dollar_value"].cummax()).min()

        # calculating the sharpe ratio for all approaches - system, one agent, baseline approach:

        # de-annualizing risk-free rate
        risk_free_rate_daily = (1 + (risk_free_rate_mean / 100)) ** (1/len(df_master)) - 1

        # getting daily percentage returns
        system_daily_returns = df_master["system_dollar_value"].pct_change().dropna()
        agent_daily_returns = df_master["agent_dollar_value"].pct_change().dropna()
        baseline_daily_returns = df_master["baseline_dollar_value"].pct_change().dropna()

        # calculating annualized sharpe ratio
        system_sharpe = (system_daily_returns.mean() - risk_free_rate_daily) / system_daily_returns.std() * np.sqrt(len(df_master))
        agent_sharpe = (agent_daily_returns.mean() - risk_free_rate_daily) / agent_daily_returns.std() * np.sqrt(len(df_master))
        baseline_sharpe = (baseline_daily_returns.mean() - risk_free_rate_daily) / baseline_daily_returns.std() * np.sqrt(len(df_master))

        # printing returns
        print(f"MAS return: {system_return * 100:.2f}%")
        print(f"{self._agent.capitalize()} Agent return: {agent_return * 100:.2f}%")
        print(f"Baseline Approach return: {baseline_return * 100:.2f}%\n")
        
        print("="* 60)

        # printing sharpe ratios
        print(f"\nMAS Sharpe Ratio: {system_sharpe:.4f}")
        print(f"{self._agent.capitalize()} Agent Sharpe Ratio: {agent_sharpe:.4f}")
        print(f"Baseline Approach Sharpe Ratio: {baseline_sharpe:.4f}\n")

        print("="* 60)

        # printing maximum drawdown
        print(f"MAS Maximum Drawdown: {system_drawdown * 100:.2f}%")
        print(f"{self._agent.capitalize()} Agent Maximum Drawdown: {agent_drawdown * 100:.2f}%")
        print(f"Baseline Approach Maximum Drawdown: {baseline_drawdown * 100:.2f}%")

        ## comparison plot of portfolios over time:

        # setting the plot
        plt.figure(figsize=(12, 6))

        # plotting system's results
        plt.plot(df_master.index, df_master["system_dollar_value"], label="MAS System", color="blue")

        # plotting single agent's results
        plt.plot(df_master.index, df_master["agent_dollar_value"], label=f"{self._agent.capitalize()} Agent", color="red")

        # plotting baseline approach
        plt.plot(df_master.index, df_master["baseline_dollar_value"], label="Baseline Approach", linestyle="--", color="green", alpha=0.7)

        # setting x-label, y-label, and title
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.title(f"Simulation Results: {self._company} {self._simulation_year}")

        # setting legend
        plt.legend(loc="best")

        # showing the plot
        plt.tight_layout()
        plt.show()

        ## agent's decisions plot:

        # getting data for all agents - predictions
        df_agents = df_master[["technical_pred", "health_pred", "semantic_pred"]]

        # setting a dictionary for values (-1 = sell, 0 = hold, 1 = buy)
        actions = {-1: "Sell", 0: "Hold", 1: "Buy"}

        # setting a dictionary for agents
        agents = {"technical_pred": "Technical", "health_pred": "Health", "semantic_pred": "Semantic"}

        # melting the dataframe - putting it to a long format
        df_long = df_agents.melt(
            value_vars=["technical_pred", "health_pred", "semantic_pred"], 
            var_name="agent",
            value_name="decision")
        
        # grouping the long datframe to obtain counts for each decision
        df_plot = df_long.groupby(["agent", "decision"]).size().reset_index(name="count")

        # mapping agents to their names
        df_plot["agent"] = df_plot["agent"].map(agents)

        # mapping decision to named labels
        df_plot["decision"] = df_plot["decision"].map(actions)

        # setting the plot
        fig, axs = plt.subplots(figsize=(12, 6))

        # plotting agents' decision and counts
        sns.barplot(data=df_plot, x="agent", y="count", hue="decision", ax=axs)

        # adding labals to the bars
        for container in axs.containers:
            axs.bar_label(container, fontsize=10, padding=3)

        # setting x-label, y-label, and title
        axs.set_xlabel("Agent")
        axs.set_ylabel("Decision Count")
        axs.set_title(f"Agents' Decisions Distribution")

        # setting legend
        axs.legend(loc="best")

        # showing the plot
        fig.tight_layout()
        plt.show()

        ## agents' confidence scores plot:

        # setting the plot
        plt.figure(figsize=(12, 6))

        # plotting technical agent's confidence scores
        plt.plot(df_master.index, df_master["technical_score"], label="Technical Agent", color="blue")

        # plotting health agent's confidence scores
        plt.plot(df_master.index, df_master["health_score"], label="Health Agent", color="red")

        # plotting semnatic agent's confidence scores
        plt.plot(df_master.index, df_master["semantic_score"], label="Semantic Agent", linestyle="--", color="green", alpha=0.7)

        # setting x-label, y-label, and title
        plt.xlabel("Date")
        plt.ylabel("Confidence Score")
        plt.title(f"Agents' Confidence Scores: {self._company} {self._simulation_year}")

        # setting legend
        plt.legend(loc="best")

        # showing the plot
        plt.tight_layout()
        plt.show()










    


