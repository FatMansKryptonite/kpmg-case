from cleaning import get_clean_data
from model import train_random_forest
from preprocessing import get_x_and_y
from evaluation import make_roc, make_shap_plots


def main():
    data = get_clean_data()

    X, y = get_x_and_y(data['data_train_fin'])
    X_train, y_train = X[:int(len(X)/2)], y[:int(len(X)/2)]
    X_test, y_test = X[int(len(X)/2):], y[int(len(X)/2):]

    model = train_random_forest(X_train, y_train)

    # Evaluate
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    make_roc(model, X_test, y_test)
    make_shap_plots(model, X_test)

    # Predict
    X_new, y_new = get_x_and_y(data['data_test_fin'])


if __name__ == '__main__':
    main()
