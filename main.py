from matplotlib import pyplot as plt

from cleaning import get_clean_data
from model import train_random_forest
from preprocessing import get_x_and_y
from evaluation import make_roc, make_shap_plots
from data_exploration import make_profiles, make_pairwise_plot


def main():
    data = get_clean_data()
    make_profiles(data)
    make_pairwise_plot(data)

    X, y = get_x_and_y(data['data_train_fin'])
    X_train, y_train = X[:int(len(X)/2)], y[:int(len(X)/2)]
    X_test, y_test = X[int(len(X)/2):], y[int(len(X)/2):]

    model = train_random_forest(X_train, y_train)

    # Evaluate
    print(f'Train score: {model.score(X_train, y_train)}')
    print(f'Test score: {model.score(X_test, y_test)}')
    make_roc(model, X_test, y_test)
    make_shap_plots(model, X_test)

    # Predict
    X_new, y_new = get_x_and_y(data['data_test_fin'])

    # Close all plots for tidyness
    plt.close('all')


if __name__ == '__main__':
    main()
