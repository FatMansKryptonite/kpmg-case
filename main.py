from cleaning import get_clean_data
from model import train_random_forest


def main():
    data = get_clean_data()

    model = train_random_forest(data['data_train_fin'])




if __name__ == '__main__':
    main()
