from datetime import datetime
import numpy as np


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
