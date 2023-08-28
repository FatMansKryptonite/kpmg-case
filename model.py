import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import json

EXCLUDE_COLS_LOCATION = "utils/exclude_cols.json"


def get_x_and_y(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    columns = df.columns
    y_col = {'missing'}

    with open(EXCLUDE_COLS_LOCATION) as file:
        exclude_cols = json.loads(file.read())
    x_cols = set(columns) - y_col - set(exclude_cols)
    x_cols = list(x_cols)
    y_col = list(y_col)

    return df[x_cols], df[y_col]


def train_random_forest(df: dict) -> RandomForestClassifier:
    X, y = get_x_and_y(df)

    model = RandomForestClassifier()
    model.fit(X, y)

    return model