"""
This file contains the different methods used to read and manipulate the autolib data
"""

import pandas as pd


def get_data(data_type: str, frac: float = 1.) -> pd.DataFrame:
    """
    :param data_type: 'train' or 'valid', tells which dataset will be returned
    :param frac: used to return a sample of the data. Indicates the fraction of the data.
    :return:
    data frame corresponding to the selected data set
    """

    if data_type == 'train':
        df = pd.read_csv('./data/ds_test_train_1M.csv', )
    elif data_type == 'eval':
        df = pd.read_csv('./data/ds_test_eval_100k.csv')
    else:
        df = pd.DataFrame()
    if frac < 1:
        df = df.sample(frac=frac, axis=0)
    return df


def get_granularity(feature: str) -> str:
    """
    returns the granularity of a feature
    :param feature: feature given in a column name
    """
    feat1 = '_'.join(feature.split('_')[1:])  # get rid of the feature_ part
    g = feat1.split('_D')[0]
    return g


def select_features(df: pd.DataFrame, type: str, granularity: str = None) -> pd.DataFrame:
    """
    select columns corresponding to type and return the restriction of df corresponding to the columns selected
    :param type: type of feature considered 'installs' or 'value'
    :param granularity: 'app', 'app_site' ...
    """
    columns = df.columns
    if granularity is None:
        select_columns = [col for col in columns if col.split('_')[-1] == type]
        return df[select_columns]

    select_columns = [col for col in columns if (col.split('_')[-1] == type) and (get_granularity(col) == granularity)]
    output = df[select_columns]
    sep = granularity + '_'
    new_names = {col: col.split(sep)[1] for col in output.columns}
    output = output.rename(columns=new_names)
    return output


def get_clean_average(df: pd.DataFrame, type: str) -> pd.DataFrame:
    """
    Used for EDA. take the dataframe with the initial data as input and return a dataframe with rows corresponding to
    the different granularities and columns corresponding to the type of figure considered.
    :param df: dataframe to analyze
    :param type: type of feature considered 'installs' or 'value'
    """
    series = []  # list of Pandas Series
    columns = df.columns
    granularities = [get_granularity(feature) for feature in columns]
    granularities = set(granularities)  # remove duplicates
    for granularity in granularities:
        granu_series = select_features(df, type, granularity).mean(axis=0)
        granu_series = granu_series.rename(granularity)
        series.append(granu_series)
    df = pd.DataFrame(series)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill the null values following the insights from the EDA.
    Revenue features will have their null values replaced by the average of the corresponding column whereas
    install features will have their null values replaced by the average of the corresponding row
    """
    # If there are rows with nan for all columns they need to be removed
    df_clean = df.copy()
    df_clean = df_clean.dropna(axis=0, how='all')

    df_install = select_features(df_clean, type='installs')
    df_revenue = select_features(df_clean, type='value')

    # build the dataframes with value replacement for install:
    install_replace = df_install.copy()
    install_values = df_install.mean(axis=1)
    for col in install_replace.columns:
        install_replace[col] = install_values

    # for revenue:
    revenue_values = df_revenue.mean(axis=0)

    df_install = df_install.fillna(value=install_replace)
    df_revenue = df_revenue.fillna(value=revenue_values)
    df_clean[df_install.columns] = df_install
    df_clean[df_revenue.columns] = df_revenue
    # in case we're missing some remaining null values
    df_clean = df_clean.fillna(0)
    return df_clean


if __name__ == '__main__':
    df = get_data('eval')
    df = preprocess(df)
    df1 = get_clean_average(df, 'value')
