from datetime import datetime
import numpy as np
from sklearn.utils import shuffle
import pandas as pd


def makeBinaryClassification(df, condition_arr, random_state=None):
    class_a = df[condition_arr == True]
    class_b = df[condition_arr == False]

    if len(class_a) == len(class_b):
        return df
    else:
        class_to_reduce = class_a if len(class_a) > len(class_b) else class_b
        class_to_keep = class_a if len(class_a) < len(class_b) else class_b

        class_reduced = shuffle(class_to_reduce, random_state=random_state, n_samples=len(class_to_keep))
        assert len(class_reduced) == len(class_to_keep)

        return shuffle(pd.concat( (class_reduced, class_to_keep) ), random_state=random_state)


def getUniqueValuesPerFeature(df, threshold=150):
    for column in df:
        print('\n' + column + ': ')
        uniques = df[column].unique()
        if len(uniques) < threshold:
            print(uniques)
        else:
            print('Too large to show')


def compareSimilarCategoricalColumns(col_a, col_b):
    ii, a_to_b = getMappingOfSimilarCategoricalColumns(col_a=col_a, col_b=col_b)
    return np.all(col_b == np.array([a_to_b[elem_a] for elem_a in col_a]))


def getMappingOfSimilarCategoricalColumns(col_a, col_b):
    cat_len = len(np.unique(col_a))
    assert cat_len == len(np.unique(col_b)), "these categorical columns do not have one to one correspondance"

    dic = dict()
    for ii, elem_a, elem_b in zip(range(len(col_a)), col_a, col_b):
        if elem_a in dic:
            assert dic[elem_a] == elem_b
        else:
            dic[elem_a] = elem_b

        if len(dic) == cat_len:
            return ii, dic


def dateStrToDayYear(dateStr):
    return datetime.strptime(dateStr, '%Y-%m-%d').timetuple().tm_yday
