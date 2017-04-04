import os
import pandas as pd


def getRbfKernelPCADimReducedDataset(n_components, target_col):
    path_data = os.path.realpath(os.path.join(os.getcwd(), '../Data',
                                              'rbf_pca_kernel_%d_components.csv' % n_components))
    assert os.path.isfile(path_data)
    df = pd.read_csv(path_data, header=0)
    XX_lowdim = df.drop(labels=[target_col], axis=1)
    yy = df[target_col]

    return XX_lowdim, yy
