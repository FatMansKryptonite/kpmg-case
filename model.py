import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier()
    model.fit(X, y)

    return model
