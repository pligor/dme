import numpy as np


class FeaturePrevDelays(object):
    """
    ## Account for previous delays from the same airport
    The algorithm goes like this:
    - get all flights from origin airport
    - go back to departure time, for some hours, but only within same day
    - sum all departure delays and divide by number of flights corresponding to that to get delay per flight
    - Make this a feature
    """

    ATTR_PREFIX = "PREV_DELAYS_"

    def __init__(self, data_frame, mins_set):
        """#note that having hours and minutes expressed like that, like an integer is not exactly
                #minutes that we extract but it is ok, we don't care to be exact"""
        # super(FeaturePrevDelays, self).__init__()
        self.df = data_frame.copy()
        self.mins_set = mins_set

    def transform(self, data_frame=None):
        df_to_transform = self.df if data_frame is None else data_frame.copy()

        for mins in self.mins_set:
            cur_key = self.ATTR_PREFIX + str(mins)
            assert cur_key not in self.df.columns
            df_to_transform[cur_key] = self.getDepDelaysPerFlight_forAllFlights(mins=mins)

        return df_to_transform

    def getDepDelaysPerFlight_forAllFlights(self, mins):
        return np.array(
            [self.getDepDelaysPerFlight_forTarget(target=flight, mins=mins) for ii, flight in self.df.iterrows()])

    def getDepDelaysPerFlight_forTarget(self, target, mins):
        same_origin = self.getSameOrigin(target=target)
        same_day_so_far = self.getSameDaySoFar(df=same_origin, target=target)
        flights_before = self.getFlightsBefore(df=same_day_so_far, target=target, mins=mins)
        flights_before_len = len(flights_before)

        if flights_before_len == 0:
            return 0
        else:
            return np.sum(flights_before['DEP_DELAY']) / flights_before_len

    @staticmethod
    def getFlightsBefore(df, target, mins):
        """#note that having hours and minutes expressed like that, like an integer is not exactly
                #minutes that we extract but it is ok, we don't care to be exact"""
        condition_mins_before = (target['DEP_TIME'] - mins) < df['DEP_TIME']
        return df[condition_mins_before]

    @staticmethod
    def getSameDaySoFar(df, target):
        condition_same_day = np.logical_and(df['MONTH'] == target['MONTH'],
                                            df['DAY_OF_MONTH'] == target['DAY_OF_MONTH'])

        # this is the original departure time
        condition_same_day_so_far = np.logical_and(condition_same_day, df['DEP_TIME'] < target['DEP_TIME'])
        return df[condition_same_day_so_far]

    def getSameOrigin(self, target):
        return self.df[self.df['ORIGIN'] == target['ORIGIN']]
