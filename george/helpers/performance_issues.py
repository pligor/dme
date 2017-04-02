from __future__ import division

import pandas as pd
from sklearn.utils import shuffle


def subsample_keeping_class_proportions(XX, yy, nn, seed=None):
    """currently works only with binary class targets"""
    assert nn % 2. == 0, "we need half and half"
    half_nn = nn // 2

    full_data = pd.concat((XX, yy), axis=1)
    yes_samples = full_data[yy].sample(half_nn, random_state=seed)
    no_samples = full_data[yy == False].sample(half_nn, random_state=seed)
    subsampled = shuffle(pd.concat((yes_samples, no_samples)), random_state=seed)

    Xsubsampled = subsampled.drop(labels=[yy.name], axis=1)
    y_subsampled = subsampled[yy.name]

    return Xsubsampled, y_subsampled
