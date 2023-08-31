import pandas as pd
from matplotlib import pyplot as plt

from cleaning import get_clean_data
from model import train_random_forest
from preprocessing import get_x_and_y
from evaluation import make_roc, make_shap_plots
from data_exploration import make_profiles, make_pairwise_plot
from sklearn.ensemble import RandomForestClassifier
from utils import balance_x_and_y, shuffle_x_and_y

SEED = 0
EXPLORE = False
EVALUATE = False


def explore(data: dict) -> None:
    make_profiles(data)
    for variable in ['missing', 'hairdresser', 'counselor', 'psychologist', 'doctor', 'dentist']:
        make_pairwise_plot(data, group_by=variable)


def evaluate(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series, name: str):
    print(f'{name} score: {model.score(X, y)}')
    make_roc(model, X, y)
    make_shap_plots(model, X)


def main():
    data = get_clean_data()

    if EXPLORE:
        explore(data)

    # Make X and y
    X, y, X_new, y_new = get_x_and_y(data)
    X_balanced, y_balanced = balance_x_and_y(X, y)
    X_balanced, y_balanced = shuffle_x_and_y(X_balanced, y_balanced, SEED)
    X_train, y_train = X_balanced[:int(len(X)/2)], y_balanced[:int(len(X)/2)]
    X_test, y_test = X_balanced[int(len(X)/2):], y_balanced[int(len(X)/2):]

    # Train model
    model = train_random_forest(X_train, y_train)

    if EVALUATE:
        evaluate(model, X_test, y_test)

    # Predict
    y_predicted = model.predict(X)
    data['data_train_fin']['missing_predicted'] = y_predicted
    data['data_train_fin'].sort_values('missing_predicted',
                                       ascending=False,
                                       inplace=True)

    # Predict new
    y_new = model.predict(X_new)
    data['data_test_fin']['missing_predicted'] = y_new
    data['data_test_fin'].sort_values('missing_predicted',
                                      ascending=False,
                                      inplace=True)

    # Close all plots for tidiness
    plt.close('all')


if __name__ == '__main__':
    main()
