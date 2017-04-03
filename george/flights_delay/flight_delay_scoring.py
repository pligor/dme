from sklearn.metrics import f1_score, precision_score
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss


def getScore_onKFold_callback(cls):
    def score_onKFold(X_train, y_train, X_valid, y_valid, i):
        cls.fit(X_train, y_train)
        preds = cls.predict(X_valid)
        proba_preds = cls.predict_proba(X_valid)
        log_loss_df = pd.DataFrame(data=np.array([[log_loss(y_true=y_valid, y_pred=proba_preds)]]),
                                   columns=["log_loss"])
        return pd.concat((getFlightDelayScores(y_true=y_valid, y_pred=preds), log_loss_df), axis=1)

    return score_onKFold


def printFlightDelayScores(y_true, y_pred):
    print "precision score for delayed flights: {}".format(precision_score(y_true=y_true, y_pred=y_pred))
    print "f1 score for both classes: {}".format(f1_score(y_true=y_true, y_pred=y_pred, average=None))
    print "f1 score for is_delayed true class: {}".format(f1_score(y_true=y_true, y_pred=y_pred))
    print "f1 score with weighted average: {}".format(f1_score(y_true=y_true, y_pred=y_pred, average='weighted'))
    print "f1 score with unweighted average: {}".format(f1_score(y_true=y_true, y_pred=y_pred, average='macro'))


def getFlightDelayScores(y_true, y_pred):
    return pd.DataFrame(data=np.array([
        precision_score(y_true=y_true, y_pred=y_pred),
        f1_score(y_true=y_true, y_pred=y_pred, average=None)[0],
        f1_score(y_true=y_true, y_pred=y_pred, average=None)[1],
        f1_score(y_true=y_true, y_pred=y_pred, average='weighted'),
        f1_score(y_true=y_true, y_pred=y_pred, average='macro'),
    ])[np.newaxis],
                        columns=["precision_score", "f1_score_nondelayed",
                                 "f1_score_delayed", "f1_score_weighted_avg",
                                 "f1_score_unweighted_avg"])
