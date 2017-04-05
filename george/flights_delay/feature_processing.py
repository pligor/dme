from sklearn.utils import shuffle
import pandas as pd
from helpers.feature_engineering import dateStrToDayYear
from helpers.my_one_hot_encoder import MyOneHotEncoder
import numpy as np


class FlightDelayFeatureProcessing(object):
    def process_all(self, df):
        df = self.createIsDelayedCol(df)
        df = self.createYday(df)
        df = self.dropYearAndDate(df)
        df = self.oneHotEncodingCarrier(df)
        df = self.oneHotEncodingOrigin(df)
        df = self.oneHotEncodingOriginCityName(df)
        df = self.oneHotEncodingOriginState(df)
        df = self.oneHotEncodingDest(df)
        df = self.oneHotEncodingDestCityName(df)
        df = self.oneHotEncodingDestState(df)
        df = self.removeCRSDeptTime(df)
        df = self.removeDepDelayNew(df)
        df = self.removeDepDel15(df)
        df = self.oneHotEncodingDepartureTimeBlock(df)
        #df = self.eraseCancelledFlights(df)
        df = self.removeCancelledAndFlights(df)
        df = self.removeElapsedTime(df)
        return self.removeArrivalAttrs(df)

    @staticmethod
    def eraseRowsWithNulls(df):
        df_clear = df.copy()
        for col in df_clear.columns:
            df_clear = df_clear[df_clear[col].isnull() == False]

        return df_clear

    @staticmethod
    def getCleanRowsWithoutAnyNulls(df):
        null_inds = set()
        for col in df.columns:
            inds = np.argwhere(df[col].isnull()).flatten()
            if len(inds) > 0:
                null_inds |= set(inds)

        return null_inds

    @staticmethod
    def getAllRowsWithAtLeastOneNull(df):
        null_inds = set()
        for col in df.columns:
            inds = np.argwhere(df[col].isnull()).flatten()
            if len(inds) > 0:
                null_inds |= set(inds)

        return null_inds

    @staticmethod
    def removeArrivalAttrs(df):
        return df.drop(labels=[
            'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP'
        ], axis=1)

    @staticmethod
    def removeElapsedTime(df):
        return df.drop(labels=['CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME'], axis=1)

    # .drop(labels=[], axis= 1)

    @staticmethod
    def eraseCancelledFlights(df):
        return df.iloc[np.argwhere(df['CANCELLED'] == False).flatten()]

    @staticmethod
    def removeCancelledAndFlights(df):
        return df.drop(labels=['CANCELLED', 'FLIGHTS'], axis=1)

    @staticmethod
    def oneHotEncodingDepartureTimeBlock(df):
        return MyOneHotEncoder().encodePandasColAndMerge(data_frame=df, col_name='DEP_TIME_BLK')

    @staticmethod
    def removeDepDel15(df):
        return df.drop(labels=['DEP_DEL15'], axis=1)

    @staticmethod
    def removeDepDelayNew(df):
        return df.drop(labels=['DEP_DELAY_NEW'], axis=1)

    @staticmethod
    def removeCRSDeptTime(df):
        return df.drop(labels=['CRS_DEP_TIME'], axis=1)

    @staticmethod
    def removeRedundantStateNames(df):
        return df.drop(labels=['DEST_STATE_NM', 'ORIGIN_STATE_NM'], axis=1)

    @staticmethod
    def oneHotEncodingDestState(df):
        return MyOneHotEncoder().encodePandasColAndMerge(data_frame=df, col_name='DEST_STATE_ABR'). \
            drop(labels=['DEST_STATE_NM'], axis=1)

    @staticmethod
    def oneHotEncodingDestCityName(df):
        return MyOneHotEncoder().encodePandasColAndMerge(data_frame=df, col_name='DEST_CITY_NAME')

    @staticmethod
    def oneHotEncodingDest(df):
        return MyOneHotEncoder().encodePandasColAndMerge(data_frame=df, col_name='DEST'). \
            drop(labels=['DEST_AIRPORT_ID'], axis=1)

    @staticmethod
    def oneHotEncodingOriginState(df):
        return MyOneHotEncoder().encodePandasColAndMerge(data_frame=df, col_name='ORIGIN_STATE_ABR'). \
            drop(labels=['ORIGIN_STATE_NM'], axis=1)

    @staticmethod
    def oneHotEncodingOriginCityName(df):
        return MyOneHotEncoder().encodePandasColAndMerge(data_frame=df, col_name='ORIGIN_CITY_NAME')

    @staticmethod
    def oneHotEncodingOrigin(df):
        return MyOneHotEncoder().encodePandasColAndMerge(data_frame=df, col_name='ORIGIN'). \
            drop(labels=['ORIGIN_AIRPORT_ID'], axis=1)

    @staticmethod
    def oneHotEncodingCarrier(df):
        return MyOneHotEncoder().encodePandasColAndMerge(data_frame=df, col_name='UNIQUE_CARRIER'). \
            drop(labels=['CARRIER'], axis=1). \
            drop(labels=['AIRLINE_ID'], axis=1)

    @staticmethod
    def removeCarrier(df):
        return df.drop(labels=['CARRIER'], axis=1)

    @staticmethod
    def removeIdColsAlreadyRepresentedByStrCols(df):
        return df.drop(labels=['ORIGIN_AIRPORT_ID', 'AIRLINE_ID', 'DEST_AIRPORT_ID'], axis=1)

    @staticmethod
    def dropYearAndDate(df):
        return df.drop(labels=['YEAR', 'FL_DATE'], axis=1)

    @staticmethod
    def createYday(df):
        df_copy = df.copy()
        df_copy['YDAY'] = [dateStrToDayYear(dateStr) for dateStr in df['FL_DATE']]
        return df_copy

    @staticmethod
    def createIsDelayedCol(df, random_state=None, shuffle_it=True):
        augmented_df = df.copy()
        augmented_df['IS_DELAYED'] = False  # all as non delayed
        augmented_df.loc[augmented_df['ARR_DELAY_GROUP'] > 0, 'IS_DELAYED'] = True  # the delayed as delayed

        if shuffle_it:
            return shuffle(augmented_df, random_state=random_state)
        else:
            return augmented_df
