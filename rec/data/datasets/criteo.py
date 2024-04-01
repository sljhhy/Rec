import pickle
import pandas as pd
import os

from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import train_test_split

from rec.data.feature_column import sparseFeature


NAMES = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
         'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
         'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
         'C23', 'C24', 'C25', 'C26']

def create_small_criteo_dataset(file, embed_dim=8, read_part=True, sample_num=100000, test_size=0.2):
    """Load small criteo data(sample num) without splitting "train.txt".
    Note: If you want to load all data in the memory, please set "read_part" to False.
    Args:
        :param file: A string. dataset's path.
        :param embed_dim: A scalar. the embedding dimension of sparse features.
        :param read_part: A boolean. whether to read part of it.
        :param sample_num: A scalar. the number of instances if read_part is True.
        :param test_size: A scalar(float). ratio of test dataset.
    :return: feature columns such as [sparseFeature1, sparseFeature2, ...],
             train, such as  ({'C1': [...], 'C2': [...]]}, [1, 0, 1, ...])
             and test ({'C1': [...], 'C2': [...]]}, [1, 0, 1, ...]).
    """
    if read_part:
        data_df = pd.read_csv(file, sep='\t', iterator=True, header=None,
                          names=NAMES)
        data_df = data_df.get_chunk(sample_num)
    else:
        data_df = pd.read_csv(file, sep='\t', header=None, names=NAMES)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)

    est = KBinsDiscretizer(n_bins=1000, encode='ordinal', strategy='uniform')
    data_df[dense_features] = est.fit_transform(data_df[dense_features])

    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    feature_columns = [sparseFeature(feat, int(data_df[feat].max()) + 1, embed_dim=embed_dim)
                        for feat in features]
    train, test = train_test_split(data_df, test_size=test_size)

    train_X = {feature: train[feature].values.astype('int32') for feature in features}
    train_y = train['label'].values.astype('int32')
    test_X = {feature: test[feature].values.astype('int32') for feature in features}
    test_y = test['label'].values.astype('int32')

    return feature_columns, (train_X, train_y), (test_X, test_y)