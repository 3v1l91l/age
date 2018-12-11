import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(train):
    excluded_features = ['target', 'user_hash',
                         # 'City_post_HOME', 'City_post_WORK',
                         # 'Raion_post_HOME', 'Raion_post_WORK',
                         # 'City_post_HOME', 'City_post_WORK',
                         'lat_quad_home', 'lat_quad_work',
                         'lon_quad_home', 'lon_quad_work',
                         'LAT_WORK', 'LAT_HOME',
                         'LON_WORK', 'LON_HOME']  # , 'data_type_3_m1', 'data_type_1_m1', 'data_type_2_m1']
    train_features = [x for x in train.columns if x not in excluded_features]

    cats = list(train.dtypes[train.dtypes == 'object'].index.values)
    cats = [x for x in cats if x not in excluded_features]

    for f in cats:
        train[f], indexer = pd.factorize(train[f])

    importances = pd.DataFrame()
    importances['feature'] = train_features
    importances['gain'] = 0
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for (train_index, valid_index) in kf.split(train, train['target']):
        trn_x, trn_y = train[train_features].iloc[train_index], train['target'].iloc[train_index]
        val_x, val_y = train[train_features].iloc[valid_index], train['target'].iloc[valid_index]
        clf = LGBMClassifier(
            objective='multiclass',

            num_class=6,
            num_leaves=16,
            max_depth=5,
            learning_rate=0.06,
            n_estimators=1000,
            subsample=.9,
            colsample_bytree=.8,
            #         lambda_l1=10,
            #         lambda_l2=0.01,
            random_state=1
        )
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            #         eval_names=['train', 'valid'],
            early_stopping_rounds=50,
            verbose=50,
            categorical_feature=cats
        )
        importances['gain'] += clf.booster_.feature_importance(importance_type='gain') / n_splits
        y_pred = clf.predict(val_x)
        acc = accuracy_score(val_y, y_pred)
        print(f'accuracy_score={acc}')
        plt.figure(figsize=(12, 16))
        sns.barplot(x='gain', y='feature', data=importances.sort_values('gain', ascending=False)[:60])
        plt.savefig('importance.png')
    #     auc = roc_auc_score(val_y, y_pred_proba[:, 1])
    #     print(f'roc auc score: {auc}')

def main():
    train = pd.read_hdf('train.hdf')
    train_model(train)

if __name__ == '__main__':
    main()