from __future__ import division

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.decomposition import KernelPCA
import os


def rbfPCAkernel(XX, yy, target_column_name, n_splits=10, n_components=2, n_jobs=1, random_state=None,
                 gamma=0.1, saving=False):
    assert n_components >= 2

    def processor(inputs):
        return KernelPCA(n_components=n_components, random_state=random_state, kernel='rbf', gamma=gamma,
                         n_jobs=n_jobs).fit_transform(inputs)

    df_lowdim = processSeparately(inputs=XX, targets=yy, processor=processor, n_components=n_components,
                                  n_splits=n_splits)

    XX_lowdim = df_lowdim.drop(labels=[target_column_name], axis=1)
    yy_lowdim = df_lowdim[target_column_name]

    if saving:
        pd.concat((XX_lowdim, yy_lowdim), axis=1).to_csv(
            os.path.realpath(os.path.join(os.getcwd(), '../Data', 'rbf_pca_kernel_%d_components.csv' % n_components)),
            index=False)

    return XX_lowdim, yy_lowdim


def subsample_keeping_class_proportions(XX, yy, nn, seed=None):
    """returns tuple of Xsub and y_sub. Currently works only with binary class targets"""
    assert nn % 2. == 0, "we need half and half"
    half_nn = nn // 2

    full_data = pd.concat((XX, yy), axis=1)
    yes_samples = full_data[yy].sample(half_nn, random_state=seed)
    no_samples = full_data[yy == False].sample(half_nn, random_state=seed)
    subsampled = shuffle(pd.concat((yes_samples, no_samples)), random_state=seed)

    Xsubsampled = subsampled.drop(labels=[yy.name], axis=1)
    y_subsampled = subsampled[yy.name]

    return Xsubsampled, y_subsampled


def processSeparately(inputs, targets, processor, n_splits=10, n_components=2, random_state=None):
    """n_components must match the number of components returned from the processor"""

    # test_data = pd.DataFrame(data=rng.rand(5,2), columns=['test', 'test1'])
    # test_tar = pd.Series(data=rng.rand(5) > 0.5, name=target_col)
    # processSeparately(inputs = test_data, targets=test_tar, processor=processor, n_components=2, n_splits=2)

    assert len(targets.shape) == 1
    target_dim = 1

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    comps_strs = ["component {}".format(ii) for ii in range(n_components)]

    proc_df = np.empty((0, n_components + target_dim))
    for _, inds in kfold.split(inputs, targets):
        curX = inputs.values[inds]
        cur_y = targets.values[inds]
        procX = processor(curX)
        if len(cur_y.shape) == 1:
            cur_y = cur_y[np.newaxis].T
        if len(procX.shape) == 1:
            procX = procX[np.newaxis].T
        # print procX.shape
        # print cur_y.shape
        proc_df = np.concatenate((
            proc_df,
            np.hstack((procX, cur_y))
        ))

    return pd.DataFrame(data=proc_df, columns=comps_strs + [targets.name])


def processSeparately_old(inputs, targets, processor, n_splits=10, n_components=2):
    """NOT working"""

    kfold = StratifiedKFold(n_splits=n_splits)

    comps_strs = ["component {}".format(ii) for ii in range(n_components)]

    proc_df = pd.DataFrame()
    for _, inds in kfold.split(inputs, targets):
        curX = inputs.iloc[inds]
        cur_y = targets.iloc[inds].to_frame()
        procX = pd.DataFrame(data=processor(curX).values, columns=comps_strs)
        proc_df = pd.concat((
            proc_df,
            pd.concat((procX, cur_y), axis=1)
        ), axis=0)

    return proc_df
