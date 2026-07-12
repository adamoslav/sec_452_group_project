from data_preparation import data_preparation
from technical_agent import technical_agent
from health_agent import health_agent
from Semantic_functions import FinancialSentimentAnalyser
import numpy as np
import pandas as pd

# Aggregation agent class
class aggregation_agent:
  def __init__(self, company, year, action_threshold=0.2):
        # Initialize with stock company and year
        self.company = company
        self.year = year
        # Define thresholds
        self.act_threshold = action_threshold    # Action threshold for weighted score decision making
        self.conf_threshold = 0.5                # High confidence threshold for conflict condition

  # Function to compute best action from average confidence score
  def get_best_action(self, avg_score):
    if avg_score > self.act_threshold:
      return 1
    elif avg_score < -self.act_threshold:
      return -1
    else:
      return 0

  # Function to compute stock advice using models aggregation
  def get_decision(self, low_weight_agent=None):
    company = self.company
    year = self.year
    conf_threshold = self.conf_threshold
    print(f"\nCompany: {company}\nYear: {year}\nAction threshold: {self.act_threshold}\nConflict high confidence threshold: {conf_threshold}\n")

    # Technical agent Simulation
    print("\n** Technical agent Simulation **")
    technical_sim = technical_agent(company, year)
    df_technical = technical_sim.model_prediction()

    print(f"\nTechnical agent prediction for company '{company}' in year {year}:\n{df_technical}\n")
    print(f"Prediction distribution:\n{df_technical['predict_action'].value_counts()}\n")

    # Health agent Simulation
    print("\n** Health agent Simulation **")
    health_sim = health_agent(company, year)
    df_health = health_sim.model_prediction()

    print(f"\nHealth agent prediction for company '{company}' in year {year}:\n{df_health}\n")
    print(f"Prediction distribution:\n{df_health['predict_action'].value_counts()}\n")

    # Sentiment agent Simulation
    print("\n** Sentiment agent Simulation **")
    sentiment_sim = FinancialSentimentAnalyser()
    df_semantic = sentiment_sim.semantic_year(company, year)
    df_semantic['semantic_pred'] = df_semantic['label'].map({'positive':1, 'neutral':0, 'negative':-1})

    print(f"\nSentiment agent prediction for company '{company}' in year {year}:\n{df_semantic}\n")
    print(f"Prediction distribution:\n{df_semantic['semantic_pred'].value_counts()}\n")

    # Filter and Rename prediction values and confidence scores for each agent
    df_tech_pred = df_technical.rename(
        columns={ 'predict_action': 'technical_pred', 'confidence_level': 'technical_score' }
      )[['technical_pred','technical_score']]
    df_health_pred = df_health.rename(
        columns={ 'predict_action': 'health_pred', 'confidence_level': 'health_score' }
      )[['health_pred','health_score']]
    df_semantic_pred = df_semantic.rename(
        columns={ 'score': 'semantic_score' }
      ).set_index('target_date')[['semantic_pred','semantic_score']]

    # Merge prediction values and confidence scores of all agents
    df_pred = pd.merge(df_tech_pred, df_health_pred, left_index=True, right_index=True)
    df_pred = pd.merge(df_pred, df_semantic_pred, left_index=True, right_index=True, how='left')

    # Prediction and Confidence score columns
    pred_cols = ['technical_pred', 'health_pred', 'semantic_pred']
    score_cols = ['technical_score', 'health_score', 'semantic_score']

    # Condition to reduce weightage of Sentiment agent in overall prediction when enabled
    # Ensure agents equal contribution when sentiment agent with dominant confidence scores
    if low_weight_agent == "sentiment":
      print("\nLow weightage enabled for Sentiment agent:")
      print("Technical agent weight = 100%\nHealth agent weight = 100%\nSentiment agent weight = 80%\n")
      df_pred['semantic_score'] *= 0.8
    df_pred[score_cols] = df_pred[score_cols].round(2)    # Roundoff confidence score by 2 decimal points
    #print(df_pred)

    # Compute normalized weighted average across all agents in range [-1,1]
    # Normalized weighted average = ∑i(prediction[i] * confidence[i]) / ∑i(confidence[i])
    weight_sum = (df_pred[pred_cols].values * df_pred[score_cols].values).sum(axis=1)
    df_pred['weight_avg'] = weight_sum / df_pred[score_cols].sum(axis=1)

    # Conflict condition - Disagreement between all agents with high confidence
    conflict_condition = (((df_pred['technical_pred'] * df_pred['health_pred'] == -1)
                            & ((df_pred['technical_score'] > conf_threshold)
                                & (df_pred['health_score'] > conf_threshold)
                                & (df_pred['semantic_score'] == 0.00))) |
                          (((df_pred[['technical_pred', 'health_pred', 'semantic_pred']].sum(axis=1) == 0)
                                & (df_pred[['technical_pred', 'health_pred', 'semantic_pred']].prod(axis=1) == 0)
                                & (df_pred[['technical_pred', 'health_pred', 'semantic_pred']].abs().sum(axis=1) == 2))
                            & ((df_pred['technical_score'] > conf_threshold)
                                & (df_pred['health_score'] > conf_threshold)
                                & (df_pred['semantic_score'] > conf_threshold))))
    print(f"\nNumber of conflict conditions across agents predictions = {conflict_condition.sum()}\n")
    #print(df_pred[conflict_condition])

    #df_pred['best_pred'] = df_pred['weight_avg'].apply(get_best_action)
    # Compute best prediction through aggregation logic
    # Fix action to 'HOLD' - 0 when conflict conditions
    df_pred['best_pred'] = np.where(conflict_condition, 0, df_pred['weight_avg'].apply(self.get_best_action))

    # Map best action values to decision labels
    df_pred['decision'] = df_pred['best_pred'].map({1:'BUY', 0:'HOLD', -1:'SELL'})
    return df_pred