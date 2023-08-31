import pandas as pd


def balance_x_and_y(X: pd.DataFrame,
                    y: pd.Series,
                    on: str = None,
                    n: int = None,
                    replacement: bool = True) -> (pd.DataFrame, pd.Series):
    if on is None:
        on = y.name

    df = X.assign(**{y.name: y})

    df = balance(df, on, n, replacement)

    X = df[X.columns]
    y = df[y.name]

    return X, y


def balance(df: pd.DataFrame,
            on: str = 'missing',
            n: int = None,
            replacement: bool = True) -> (pd.DataFrame, pd.Series):

    if n is None:
        if replacement:
            n = df[on].value_counts().values.max()  # Up-sample
        else:
            n = df[on].value_counts().values.min()  # Down-sample
    elif n == 'original':
        n = len(df)

    g = df.groupby(on)
    df = g.apply(lambda x: x.sample(n, replace=replacement).reset_index(drop=True))

    return df.reset_index(drop=True)


def shuffle_x_and_y(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> (pd.DataFrame, pd.Series):
    df = X.assign(**{y.name: y})

    df = shuffle(df, seed)

    X = df[X.columns]
    y = df[y.name]

    return X, y


def shuffle(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    return df.sample(frac=1, random_state=seed)
