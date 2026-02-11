import pandas as pd, numpy as np, seaborn as sns, torch as tc, torch.nn as nn, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


def clean_database(origin_path):

    file_path = os.path.join(origin_path, "breast-cancer.csv")

    return pd.read_csv(file_path, sep=',', encoding='iso-8859-1')


if __name__ == '__main__':

    np.random.seed(1)

    tc.manual_seed(1)

    file_dir = os.path.dirname(os.path.abspath(__file__))

    db = clean_database(file_dir)

    db = db.loc[:, ~db.columns.str.contains('^Unnamed')]

    db['diagnosis'].replace({
        'M': 1,
        'B': 0
    }, inplace=True)

    classes = db.iloc[:, 1]

    entrys = db.iloc[:, 2:32]

    entrys = StandardScaler().fit_transform(entrys)








