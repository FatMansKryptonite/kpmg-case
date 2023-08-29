from cleaning import get_clean_data
from model import train_random_forest
from preprocessing import get_x_and_y


def main():
    data = get_clean_data()

    X_train, y_train = get_x_and_y(data['data_train_fin'])
    X_test, y_test = get_x_and_y(data['data_test_fin'])
    model = train_random_forest(X_train[:int(len(X_train)/2)], y_train[:int(len(X_train)/2)])

    comparison = abs(y_train[int(len(X_train)/2):].to_numpy() - model.predict(X_train[int(len(X_train)/2):]))


if __name__ == '__main__':
    main()
