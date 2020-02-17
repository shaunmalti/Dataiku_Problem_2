import numpy as np
import pandas as pd
import seaborn as sns

import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
pd.set_option('expand_frame_repr', False)

def importData():
    train = pd.read_csv('./us_census_full/census_income_learn.csv')
    test = pd.read_csv('./us_census_full/census_income_test.csv')
    return train, test

def explore(data):
    summaryDf = pd.DataFrame(data.dtypes, columns=['dtypes'])
    summaryDf = summaryDf.reset_index()
    summaryDf['Name'] = summaryDf['index']
    summaryDf['Missing'] = data.isnull().sum().values
    summaryDf['Total'] = data.count().values
    summaryDf['MissPerc'] = (summaryDf['Missing']/data.shape[0])*100
    summaryDf['NumUnique'] = data.nunique().values
    summaryDf['UniqueVals'] = [data[col].unique() for col in data.columns]
    print(summaryDf.head(50))


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def import_data(name):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv('us_census_full/' + name + '.csv', parse_dates=True, keep_date_col=True)
    # df = reduce_mem_usage(df)
    return df

def shapRanking(x_data, y_data):
    train_x, valid_x, train_y, valid_y = train_test_split(
        x_data, y_data, test_size=0.33, random_state=42)

    train_data = lgb.Dataset(train_x, label=train_y)
    valid_data = lgb.Dataset(valid_x, label=valid_y)

    # LGB parameters:
    params = {'learning_rate': 0.05,
              'boosting': 'gbdt',
              'objective': 'binary',
              'num_leaves': 2000,
              'min_data_in_leaf': 200,
              'max_bin': 200,
              'max_depth': 16,
              'seed': 2018,
              'nthread': 10, }

    # LGB training:
    lgb_model = lgb.train(params, train_data,
                          num_boost_round=2000,
                          valid_sets=(valid_data,),
                          verbose_eval=10,
                          early_stopping_rounds=20)

    explainer = shap.TreeExplainer(lgb_model).shap_values(valid_x)
    shap.summary_plot(explainer[1], valid_x)
    plt.show()

    shap.dependence_plot('Wage', explainer[1], valid_x)
    plt.show()

def weighted_hist(x, weights, **kwargs):
    plt.hist(x, weights=weights, **kwargs)
    plt.xticks(rotation=90)

def plots(data):
    grid = sns.FacetGrid(data, col='Target', aspect=1.6)
    grid.map(weighted_hist, 'LabourUnion', 'InstanceWeight', bins=np.arange(data['LabourUnion'].nunique())-0.5)
    plt.show()

    # labourUnionDF = data.loc[
    #     (data['LabourUnion'] == ' Not in universe') & (data['Age'] > 18) & (data['Age'] < 65)]
    # grid = sns.FacetGrid(labourUnionDF, col='Target', aspect=1.6)
    # grid.map(weighted_hist, 'IndustryCodeString', 'InstanceWeight',
    #          bins=np.arange(data['IndustryCodeString'].nunique()) - 0.5)
    # plt.show()

def main():
    train = import_data('census_income_learn')
    test = import_data('census_income_test')

    # train_target = train['Target']
    # train.drop(['Target'], axis=1, inplace=True)

    # test_target = test['Target']
    # test.drop(['Target'], axis=1, inplace=True)

    df = pd.concat([train, test])

    # df = df.replace([' Not in universe'], [None])
    # df = df.replace([' ?'], [None])

    plots(df)
    exit()

    # not in universe or children in column IndustryCodeString - make the distinction, if age is <=16 therefore children
    # else not in universe
    df.drop(['IndustryCodeString', 'OccupationCodeString'], axis=1, inplace=True)
    # children group
    df.loc[(df['IndustryCode'] == 0) & (df['Age'] <= 18), 'IndustryCode'] = 52
    # people not in any industry group
    df.loc[(df['IndustryCode'] == 0) & (df['Age'] > 18) & (df['Age'] < 65), 'IndustryCode'] = 53
    # old people group
    df.loc[(df['IndustryCode'] == 0) & (df['Age'] >= 65), 'IndustryCode'] = 54

    # latino is classed as being from (chicano is someone from mexico) cuba, mexico, puerto rico,
    # south or central america, or other spanish culture or origin regardless of race
    # df.loc[df['HispOrig'] == ' Chicano', 'HispOrig'] = ' Mexican (Mexicano)'
    df.loc[
        df['HispOrig'].isin([' Chicano', ' Cuban', ' Other Spanish', ' Puerto Rican', ' Central or South American',
                             ' Mexican (Mexicano)', ' Other Spanish']),
        'HispOrig'
    ] = ' Latino'

    df.loc[df['HispOrig'] == ' Do not know', 'HispOrig'] = ' NA'

    # altering education categories
    df.loc[
        (df['Education'].isin(
            [' 7th and 8th grade', ' 10th grade', ' 11th grade', ' 9th grade', ' 12th grade no diploma']
        )) & (df['Age'] > 18), 'Education'
    ] = ' High School Dropout'

    df.loc[
        (df['Education'].isin(
            [' 7th and 8th grade', ' 10th grade', ' 11th grade', ' 9th grade', ' 12th grade no diploma']
        )) & (df['Age'] <= 18), 'Education'
    ] = ' High School'
    df.loc[(df['Education'].isin([' 5th or 6th grade', ' 1st 2nd 3rd or 4th grade'])) & (
                df['Age'] >= 12), 'Education'] = ' Elementary School Dropout'
    df.loc[(df['Education'].isin([' 5th or 6th grade', ' 1st 2nd 3rd or 4th grade'])) & (
                df['Age'] < 12), 'Education'] = ' Elementary School'
    df.loc[(df['Education'] == ' Less than 1st grade') & (df['Age'] > 7), 'Education'] = ' No Education'

    # altering HouseLive1Yr category
    df.loc[df['HouseLive1Yr'] == ' Not in universe under 1 year old', 'HouseLive1Yr'] = ' No'

    # altering household summary stat
    df.loc[df[
               'HouseholdSummaryStat'] == ' Child under 18 ever married', 'HouseholdSummaryStat'] = ' Child under 18 never married'

    explore(df)
    return

if __name__ == '__main__':
    main()