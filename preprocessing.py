import pandas as pd
import json

EXCLUDE_COLS_LOCATION = "utils/preprocessing_steps.json"

with open(EXCLUDE_COLS_LOCATION) as file:
    preprocessing_instructions = json.loads(file.read())


def select_cols(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    columns = df.columns
    y_col = {'missing'}

    exclude_cols = preprocessing_instructions['exclude']
    x_cols = set(columns) - y_col - set(exclude_cols)
    x_cols = list(x_cols)
    y_col = list(y_col)[0]

    return df[x_cols], df[y_col]


def one_hot_encode_x(X: pd.DataFrame) -> pd.DataFrame:
    one_hot_encode_cols = preprocessing_instructions['one_hot_encode']

    X = pd.get_dummies(X,
                       prefix=one_hot_encode_cols,
                       columns=one_hot_encode_cols,
                       drop_first=True,
                       dtype=int)

    return X


def convert_to_numerical(X: pd.DataFrame) -> pd.DataFrame:
    to_numerical_cols = preprocessing_instructions['to_numerical']
    X[to_numerical_cols] = X[to_numerical_cols].astype('int64')

    return X


def get_x_and_y(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    X, y = select_cols(df)
    X = one_hot_encode_x(X)
    X = convert_to_numerical(X)

    return X, y
