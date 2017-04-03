from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


class MyCrossValidation(object):
    def __init__(self, n_folds=5, random_state=None):
        # super(MyCrossValidation, self).__init__()
        self.kFold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    def onEachKFold(self, inputs, targets, action, take_average=True):
        i = 0
        results = pd.DataFrame()
        for curTrainIndices, curTestingIndices in self.kFold.split(inputs):
            curXtrain = inputs.iloc[curTrainIndices]
            curYtrain = targets[curTrainIndices]

            curXvalid = inputs.iloc[curTestingIndices]
            curYvalid = targets[curTestingIndices]

            results = pd.concat((
                results, action(curXtrain, curYtrain, curXvalid, curYvalid, i)
            ))

            i += 1

        if take_average:
            return np.mean(results, axis=0)  # returns panda series
        else:
            return results  # returns dataframe
