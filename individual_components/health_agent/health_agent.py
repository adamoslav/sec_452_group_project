##imports
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class health_agent:
    def __init__(self, company, year):
        # store inputs
        self.company = company
        self.year = year

        # importing functions created in data prep
        from data_preparation import data_preparation

        # activate data preparation class
        self.activate_class = data_preparation(self.company, self.year)

        # gets the training data
        self.train_data = self.activate_class.train_data_health_agent()

    def model_prediction(self):
        # split training data into:
        #features (x) financial health indicators inc. revenue_growth, net_income, profit_margin, debt_to_equity_ratio, current_ratio, EPS, price_to_earnings_ratio
        # labels (y) action = buy, sell or hold
        x_train = self.train_data.drop(columns=["action"])
        y_train = self.train_data["action"]

        # create model
        model = RandomForestClassifier(n_estimators=500, random_state=42)

        #train model on training split
        model.fit(x_train, y_train)

        # get simulation data (no actions)
        test_data = self.activate_class.simulation_data_health_agent()

        pred_prob = model.predict_proba(test_data)
        df_prob = pd.DataFrame(pred_prob, columns=['Sell_prob','Hold_prob','Buy_prob'], index=test_data.index)
        # store prediction in dataframe
        df_prob['predict_action'] = model.predict(test_data)
        df_prob['confidence_level'] = pred_prob.max(axis=1)
        
        return df_prob