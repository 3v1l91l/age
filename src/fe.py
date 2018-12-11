import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import os
import numpy as np

def mean_encode_cat(df, cols, target_col='target'):
    print('mean_encode_cat')
    for col in cols:
        print(col)
        for target_class in np.array(range(6))+1:
            temp = df[[target_col, col]]
            temp[target_col] = temp[target_col] == target_class
            gr = temp.groupby(col)[target_col].mean()
            gr.name = gr.name + '_mean_enc_' + str(target_class)
            df = df.merge(gr.reset_index(), how='left', right_on=col, left_on=col)
        df.drop(col, axis=1, inplace=True)
    return df


def feature_engineering(df):
    df['City_post_HOME_same_WORK'] = df['City_post_HOME'] != df['City_post_WORK']
    df['Raion_post_HOME_same_WORK'] = df['Raion_post_HOME'] != df['Raion_post_WORK']
    df['Oblast_post_HOME_same_WORK'] = df['Oblast_post_HOME'] != df['Oblast_post_WORK']
    # df['lat_quad_HOME_same_WORK'] = df['lat_quad_home'] != df['lat_quad_work']
    # df['lon_quad_HOME_same_WORK'] = df['lon_quad_home'] != df['lon_quad_work']
    cat_cols = ['device_brand', 'device_model', 'software_os_vendor', 'software_os_name', 'software_os_version',]
    mean_encode_cat(df, cat_cols, target_col='target')

    volume_cols = [x for x in df.columns if x.endswith('_volume')]
    for col in volume_cols:
        df[col + '_volume_div_DATA_VOLUME_WEEKDAYS'] = df[col] / df['DATA_VOLUME_WEEKDAYS']
        df[col + '_volume_div_DATA_VOLUME_WEEKENDS'] = df[col] / df['DATA_VOLUME_WEEKENDS']

    return df


def main():
    train = pd.read_csv(os.path.join('..', 'input', 'train_age.csv'))#, usecols=range(324))
    train = feature_engineering(train)
    train.to_hdf('train.hdf', 'train')

if __name__ == '__main__':
    main()