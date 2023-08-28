import numpy as np
import pandas as pd
import os
from ydata_profiling import ProfileReport

DATA_FOLDER = 'ml_test_case_martians'


def get_data() -> dict:
    data = {}
    for file_path in os.listdir(DATA_FOLDER):
        data[os.path.splitext(file_path)[0]] = pd.read_csv(os.path.join(DATA_FOLDER, file_path))

    return data


def add_century_info(year: pd.Series) -> pd.Series:
    # TODO Does not handle case of people born after year 2200.
    year = str(int(year) + 2100)

    return year


def convert_date_format(s: pd.Series) -> pd.Series:
    date_info = s.str.split('/', expand=True)

    year = date_info[2].apply(add_century_info)
    month = date_info[0]
    day = date_info[1]

    return pd.concat([year, month, day], axis=1).agg('-'.join, axis=1)


def clean_data_fin(df: pd.DataFrame) -> pd.DataFrame:
    df['date_of_birth'] = convert_date_format(df['date_of_birth'])

    for key in ['date_of_birth', 'last_seen']:
        df[key] = pd.to_datetime(df[key])

    return df


def clean_data(data: dict) -> dict:
    for key in data.keys():
        data[key] = data[key].replace({'nan': np.nan})
        data[key] = data[key].dropna()

        if key in ['data_test_fin', 'data_train_fin']:
            data[key] = clean_data_fin(data[key])
        else:
            pass

    return data


def merge_missing(data: dict) -> dict:
    missing_df = data['missing_report']
    missing_df['full_name'] = missing_df[['first_name', 'last_name']].agg(' '.join, axis=1)

    for key in ['data_train_fin', 'data_test_fin']:
        df = data['data_train_fin'].merge(missing_df[['full_name', 'missing']],
                                                              how='left',
                                                              on='full_name')
        df.fillna(0, inplace=True)
        data[key] = df

    del data['missing_report']

    return data


def make_profiles(data: dict, location: str = 'profiles') -> None:
    for key in data.keys():
        prof = ProfileReport(data[key])
        prof.to_file(output_file=f'{location}/{key}.html')


def get_clean_data(generate_profiles: bool = False) -> dict:
    data = get_data()
    data = clean_data(data)
    data = merge_missing(data)

    if generate_profiles:
        make_profiles(data)

    return data
