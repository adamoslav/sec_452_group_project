# imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

from data_preparation import data_preparation


def health_agent(company="Apple", year=2024):
    # activate data preparation class
    activate_class = data_preparation(company, year)

    # get the training data
    train_data = activate_class.train_data_health_agent()

    # split training data into:
    # features (x) financial health indicators inc. revenue_growth, net_income, profit_margin, debt_to_equity_ratio, current_ratio, EPS, price_to_earnings_ratio
    # labels (y) action = buy, sell or hold
    x_train = train_data.drop(columns=["action"])
    y_train = train_data["action"]

    # create model
    model = RandomForestClassifier(n_estimators=500, random_state=42)

    # split data into training and testing sets
    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    # train model on test split
    model.fit(x_train_split, y_train_split)

    # evaluate model on test split
    y_pred = model.predict(x_test_split)

    accuracy = accuracy_score(y_test_split, y_pred)
    print("Health Agent Accuracy:", accuracy)

    classification = classification_report(y_test_split, y_pred)
    print("Classification Report:\n", classification)

    cm = confusion_matrix(y_test_split, y_pred)
    print("Confusion Matrix:\n", cm)

    # get simulation data (no actions)
    test_data = activate_class.simulation_data_health_agent()

    # predict probabilities and actions
    pred_prob = model.predict_proba(test_data)
    pred_action = model.predict(test_data)

    # get order of class labels
    classes = model.classes_

    # create output dataframe
    #includes dates and features
    df = pd.DataFrame(index=test_data.index)

    # loop through the class labels and assign each probability
    # column to the correct output column:
    # Sell_prob, Hold_prob, or Buy_prob
    for action in range(len(classes)):
        cls = classes[action]
        # assign probabilities to correct columns
        if cls == -1:
            df["Sell_prob"] = pred_prob[:, action]
        elif cls == 0:
            df["Hold_prob"] = pred_prob[:, action]
        elif cls == 1:
            df["Buy_prob"] = pred_prob[:, action]

    #checks
    # make sure all columns exist and if not 0
    for col in ["Sell_prob", "Hold_prob", "Buy_prob"]:
        if col not in df.columns:
            df[col] = 0.0

    # order columns
    df = df[["Sell_prob", "Hold_prob", "Buy_prob"]]

    # add prediction and confidence
    df["predict_action"] = pred_action
    df["confidence_level"] = pred_prob.max(axis=1)

    return df


# example run for Apple
result = health_agent("Apple", 2024)
print(result)