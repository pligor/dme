import numpy as np


class FeatureAirlineDelays(object):
    """
    ## Account delays of same airline
    Algorithm:
    - keep trips where destination is origin's target
    - keep trips only of the same day
    - keep trips where the departure time is target's departure time - CRS elapsed time (even if they just arrived)
    - go further back in time within the same day and consume all departure delays as above
        - sum dep delays and divide  by number of flights
    """

    ATTR_PREFIX = "AIRLINE_DELAYS_"

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
        cond1 = np.logical_and(self.condDestinationTargetsOrigin(target=target),
                               self.condSameAirline(target=target))
        cond2 = np.logical_and(cond1, self.condSameDay(target=target))

        shouldHaveArrivedFlights = self.getShouldHaveArrivedFlights(df_arriving_same_day_airline=self.df[cond2],
                                                                    target=target)

        flights_before = self.getFlightsBefore(df=shouldHaveArrivedFlights, target=target, mins=mins)

        flights_before_len = len(flights_before)

        if flights_before_len == 0:
            return 0
        else:
            return np.sum(flights_before['DEP_DELAY']) / flights_before_len

    @staticmethod
    def getFlightsBefore(df, target, mins):
        """#note that having hours and minutes expressed like that, like an integer is not exactly
                #minutes that we extract but it is ok, we don't care to be exact"""
        shouldHaveDeparted = target['DEP_TIME'] - df['CRS_ELAPSED_TIME']

        condition_mins_before = (shouldHaveDeparted - mins) < df['DEP_TIME']
        return df[condition_mins_before]

    @staticmethod
    def getShouldHaveArrivedFlights(df_arriving_same_day_airline, target):
        shouldHaveDeparted = target['DEP_TIME'] - df_arriving_same_day_airline['CRS_ELAPSED_TIME']

        cond = df_arriving_same_day_airline['DEP_TIME'] < shouldHaveDeparted

        return df_arriving_same_day_airline[cond]

    def condSameDay(self, target):
        return np.logical_and(self.df['MONTH'] == target['MONTH'],
                              self.df['DAY_OF_MONTH'] == target['DAY_OF_MONTH'])

    def condSameAirline(self, target):
        return self.df['UNIQUE_CARRIER'] == target['UNIQUE_CARRIER']

    def condDestinationTargetsOrigin(self, target):
        return self.df['DEST'] == target['ORIGIN']
