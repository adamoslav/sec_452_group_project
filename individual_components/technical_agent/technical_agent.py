# create a class for technical agent
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# importing the class
class technical_agent:
    def __init__(self, company, year):
        # initial variable
        self.company = company
        self.year = year
        # import data preparation class
        from data_preparation import data_preparation
        print("Class imported successfully!")
        self.activate_class = data_preparation(self.company, self.year)
        print("Class activated!")

        # technical agent's training data
        self.train_data = self.activate_class.train_data_technical_agent()

    # model training
    def model_prediction(self):
        # drop label
        X_train = self.train_data.drop(columns=['action'])
        Y_train = self.train_data['action']

        #initial random forest model
        tech_model = RandomForestClassifier(n_estimators=500, random_state=42)
        # train model
        tech_model.fit(X_train, Y_train)

        # get simulated data
        sim_data = self.activate_class.simulation_data_technical_agent()
        # make prediction
        pred_prob = tech_model.predict_proba(sim_data)
        df_prob = pd.DataFrame(pred_prob, columns=['Sell_prob','Hold_prob','Buy_prob'], index=sim_data.index)
        # store prediction in dataframe
        df_prob['predict_action'] = tech_model.predict(sim_data)
        df_prob['confidence_level'] = pred_prob.max(axis=1)
        
        return df_prob
