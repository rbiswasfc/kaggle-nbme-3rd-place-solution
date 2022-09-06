import pandas as pd
from sklearn.model_selection import GroupKFold


def make_fold(df):
    df.loc[:, 'fold'] = -1
    gkf = GroupKFold(n_splits=5)
    for n, (_, valid_index) in enumerate(gkf.split(df['id'], df['location'], df['pn_num'])):
        df.loc[valid_index, 'kfold'] = n
    return df


train_df = pd.read_csv("../data/train_data/train.csv")
train_df = make_fold(train_df)
train_df.to_csv("../data/train_data/train_with_folds.csv", index=False)
