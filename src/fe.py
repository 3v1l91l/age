import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import os

def feature_engineering(df):
    df['City_post_HOME_same_WORK'] = df['City_post_HOME'] != df['City_post_WORK']
    df['Raion_post_HOME_same_WORK'] = df['Raion_post_HOME'] != df['Raion_post_WORK']
    df['Oblast_post_HOME_same_WORK'] = df['Oblast_post_HOME'] != df['Oblast_post_WORK']

    return df


def main():
    train = pd.read_csv(os.path.join('..', 'input', 'train_age.csv'), usecols=range(324))
    train = feature_engineering(train)
    train.to_hdf('train.hdf', 'train')

if __name__ == '__main__':
    main()